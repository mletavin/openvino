// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/constant.hpp"

#include "api/convolution.hpp"
#include "api/deconvolution.hpp"
#include "api/binary_convolution.hpp"
#include "api/reshape.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {


struct ConvoltuionParameters {
    cldnn::tensor stride;
    cldnn::tensor padding;
    cldnn::tensor dilation;
    uint32_t groups;
};

static ConvoltuionParameters GetConvolutionParameters(const ngraph::CoordinateDiff& pads_begin,
                                                      const ngraph::Strides& dilations,
                                                      const ngraph::Strides& strides,
                                                      uint32_t groups) {
    cldnn::tensor stride, padding, dilation;
    if (pads_begin.size() != strides.size() || dilations.size() != strides.size())
        THROW_IE_EXCEPTION << "Strides, Dilations and Pads are supposed to have the same elements count";

    switch (strides.size()) {
        case 3: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[2], strides[1], strides[0]));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[2], -pads_begin[1], -pads_begin[0]));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[2], dilations[1], dilations[0]));
            break;
        }
        case 2: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[1], strides[0], 1));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[1], -pads_begin[0], 0));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[1], dilations[0], 1));
            break;
        }
        case 1: {
            stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[0], 1, 1));
            padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pads_begin[0], 0, 0));
            dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[0], 1, 1));
            break;
        }
        default: THROW_IE_EXCEPTION << "Unsupported convolve parameters size. Only 1d, 2d, and 3d cases are supported";
    }

    return {stride, padding, dilation, groups};
}

void Program::CreateGroupConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolution>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t groups = op->get_input_shape(1)[0];
    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), groups);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    auto weightsName = inputs[1];
    // WA: For non-constant weights (such as FakeQuantize op) dimensions order is GOIYZ, but
    // the selected format is OIZYX by default.
    if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node->get_input_node_shared_ptr(1)) == nullptr) {
        std::string reshapeName = layerName + "_cldnn_weights_reshape";
        std::string reorderName = layerName + "_cldnn_weights_reorder";

        auto weights_shape = op->get_input_shape(1);
        std::vector<size_t> new_weights_shape;
        new_weights_shape.push_back(weights_shape[0] * weights_shape[1]); // Merged G and O dims
        for (size_t i = 2; i < weights_shape.size(); i++) {
            new_weights_shape.push_back(weights_shape[i]);
        }
        auto reshapePrim = cldnn::reshape(reshapeName,
                                          weightsName,
                                          CldnnTensorFromIEDims(new_weights_shape));

        topology.add(reshapePrim);
        AddInnerPrimitiveToProfiler(reshapeName, layerName, node);

        auto reorderPrim = cldnn::reorder(reorderName,
                                          reshapeName,
                                          DefaultFormatForDims(new_weights_shape.size()),
                                          DataTypeFromPrecision(op->get_input_element_type(1)));

        topology.add(reorderPrim);
        AddInnerPrimitiveToProfiler(reorderName, layerName, node);

        weightsName = reorderName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};
    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       params.groups,
                                       params.stride,
                                       params.padding,
                                       params.dilation,
                                       CldnnTensorFromIEDims(outDims),
                                       DataTypeFromPrecision(outPrecision));

    topology.add(convPrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Convolution>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    auto convPrim = cldnn::convolution(layerName,
                                       inputs[0],
                                       weights,
                                       {},
                                       params.groups,
                                       params.stride,
                                       params.padding,
                                       params.dilation,
                                       CldnnTensorFromIEDims(outDims),
                                       DataTypeFromPrecision(outPrecision));

    topology.add(convPrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateConvolutionBackpropDataOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ConvolutionBackpropData>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    // 3rd input is an optional output shape
    ValidateInputs(op, {2, 3});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            THROW_IE_EXCEPTION << "Unsupported dilation in ConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);

    auto weightsName = inputs[1];
    // WA: For non-constant weights (such as FakeQuantize op) dimensions order is IOYX, but
    // the selected format is OIYX by default. So we need to swap I and O dimensions to match the format
    if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node->get_input_node_shared_ptr(1)) == nullptr) {
        std::string reshapeName = layerName + "_cldnn_weights_reshape";

        auto weights_shape = op->get_input_shape(1);
        std::swap(weights_shape[0], weights_shape[1]);
        auto reshapePrim = cldnn::reshape(reshapeName,
                                          weightsName,
                                          CldnnTensorFromIEDims(weights_shape));

        topology.add(reshapePrim);
        AddInnerPrimitiveToProfiler(reshapeName, layerName, node);

        weightsName = reshapeName;
    }

    std::vector<cldnn::primitive_id> weights = {weightsName};
    auto deconvPrim = cldnn::deconvolution(layerName,
        inputs[0],
        weights,
        {},
        params.groups,
        params.stride,
        params.padding,
        CldnnTensorFromIEDims(op->get_output_tensor(0).get_shape()));
    topology.add(deconvPrim);

    AddPrimitiveToProfiler(op);
}

void Program::CreateGroupConvolutionBackpropDataOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolutionBackpropData>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto dilations = op->get_dilations();
    for (auto d : dilations) {
        if (d != 1) {
            THROW_IE_EXCEPTION << "Unsupported dilation in ConvolutionBackpropData " << op->get_friendly_name();
        }
    }

    uint32_t groups = op->get_input_shape(1)[0];
    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), groups);
    auto weightsName = inputs[1];
    // WA: For non-constant weights (such as FakeQuantize op) dimensions order is GIOYX, but
    // the selected format is GOIYX by default. So we need to swap I and O dimensions to match the format
    if (std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node->get_input_node_shared_ptr(1)) == nullptr) {
        std::string reshapeName = layerName + "_cldnn_weights_reshape";

        auto weights_shape = op->get_input_shape(1);
        std::swap(weights_shape[1], weights_shape[2]);
        auto reshapePrim = cldnn::reshape(reshapeName,
                                          weightsName,
                                          CldnnTensorFromIEDims(weights_shape));

        topology.add(reshapePrim);
        AddInnerPrimitiveToProfiler(reshapeName, layerName, node);

        weightsName = reshapeName;
    }
    std::vector<cldnn::primitive_id> weights = {weightsName};

    auto deconvPrim = cldnn::deconvolution(layerName,
        inputs[0],
        weights,
        {},
        params.groups,
        params.stride,
        params.padding,
        CldnnTensorFromIEDims(op->get_output_tensor(0).get_shape()));
    topology.add(deconvPrim);

    AddPrimitiveToProfiler(op);
}

void Program::CreateDeformableConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::DeformableConvolution>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {3});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), op->get_group());
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[2]};
    if (params.groups > 1) {
        auto convPrim = cldnn::convolution(layerName,
                                           inputs[0],
                                           inputs[1],
                                           weights,
                                           {},
                                           params.groups,
                                           op->get_deformable_group(),
                                           params.stride,
                                           params.padding,
                                           params.dilation,
                                           CldnnTensorFromIEDims(outDims));

        topology.add(convPrim);
        AddPrimitiveToProfiler(op);
    } else {
        std::string defConvLayerNameInterp = layerName + "_interp";
        std::string defConvLayerNameConv = layerName;
        cldnn::tensor kernel;
        auto weights_shape = op->get_input_shape(2);
        size_t sidx = 2 + (params.groups > 1 ? 1 : 0);
        if (weights_shape.size() == 3) {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[sidx + 2],
                                                  weights_shape[sidx + 1],
                                                  weights_shape[sidx + 0]));
        } else {
            kernel = cldnn::tensor(cldnn::batch(1),
                                   cldnn::feature(1),
                                   cldnn::spatial(weights_shape[sidx + 1],
                                                  weights_shape[sidx + 0],
                                                  1));
        }

        auto defConvPrimInterp = cldnn::deformable_interp(defConvLayerNameInterp,
                                                          inputs[0],
                                                          inputs[1],
                                                          params.groups,
                                                          op->get_deformable_group(),
                                                          params.stride,
                                                          params.padding,
                                                          params.dilation,
                                                          CldnnTensorFromIEDims(outDims),
                                                          kernel);
        topology.add(defConvPrimInterp);
        AddInnerPrimitiveToProfiler(defConvLayerNameInterp, defConvLayerNameConv, op);
        auto defConvPrim = cldnn::deformable_conv(defConvLayerNameConv,
                                                  defConvLayerNameInterp,
                                                  weights,
                                                  {},
                                                  params.groups,
                                                  CldnnTensorFromIEDims(outDims));
        topology.add(defConvPrim);
        AddPrimitiveToProfiler(defConvLayerNameConv, op);
    }
}

void Program::CreateBinaryConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::BinaryConvolution>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetConvolutionParameters(op->get_pads_begin(), op->get_dilations(), op->get_strides(), 1);
    auto outDims = op->get_output_shape(0);
    auto outPrecision = op->get_output_element_type(0);

    std::vector<cldnn::primitive_id> weights = {inputs[1]};
    cldnn::data_types calc_precision = DataTypeFromPrecision(op->get_output_element_type(0));
    auto convPrim = cldnn::binary_convolution(layerName,
                                              inputs[0],
                                              weights,
                                              params.stride,
                                              params.padding,
                                              params.dilation,
                                              CldnnTensorFromIEDims(outDims),
                                              params.groups,
                                              op->get_pad_value(),
                                              calc_precision);

    topology.add(convPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
