// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/reshape.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

void Program::CreateCommonReshapeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op) {
    ValidateInputs(op, {1, 2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto inDims = op->get_input_shape(0);
    auto outDims = op->get_output_shape(0);
    auto outTensor = CldnnTensorFromIEDims(outDims);

    // if we convert from or to 5D/6D, additional reorder also required to change format
    cldnn::primitive_id reshapeInputId = inputPrimitives[0];
    if (inDims.size() != outDims.size()) {
        cldnn::primitive_id reorderId = "reorder:" + op->get_friendly_name() + "_reorder";
        cldnn::format outputFormat = cldnn::format::bfyx;

        switch (outDims.size()) {
        case 5: outputFormat = cldnn::format::bfzyx; break;
        case 6: outputFormat = cldnn::format::bfwzyx; break;
        default: break;
        }

        cldnn::layout outputLayout(DataTypeFromPrecision(op->get_output_element_type(0)), outputFormat, outTensor);
        topology.add(cldnn::reorder(reorderId, reshapeInputId, outputLayout));
        InitProfileInfo(reorderId, "Reorder", false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, layerName);
        primitivesToIRLayersMap[reorderId] = { op->get_friendly_name() };
        primitiveIDs[layerName + "_reorder"] = reorderId;
        primitiveIDs[reorderId] = reorderId;
        profilingIDs.push_back(reorderId);
        reshapeInputId = reorderId;
    }

    auto reshapePrim = cldnn::reshape(layerName,
                                      reshapeInputId,
                                      outTensor);

    topology.add(reshapePrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateReshapeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateCommonReshapeOp(topology, op);
}

void Program::CreateSqueezeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateCommonReshapeOp(topology, op);
}

void Program::CreateUnsqueezeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateCommonReshapeOp(topology, op);
}

}  // namespace CLDNNPlugin
