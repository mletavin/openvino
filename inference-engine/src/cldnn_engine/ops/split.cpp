// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/crop.hpp"

namespace CLDNNPlugin {


void Program::CreateCommonSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op) {
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto inputDims = op->get_input_shape(0);
    InferenceEngine::SizeVector startOffset(inputDims.size());

    bool is_single_out_split = op->get_output_size() == 1;

    for (size_t i = 0; i < op->get_output_size(); i++) {
        std::string outLayerName = layerName + (is_single_out_split ? "" : "." + std::to_string(i));
        const auto outLayerDims = op->get_output_shape(i);
        if (outLayerDims.size() != startOffset.size()) {
            THROW_IE_EXCEPTION << "Invalid dimesions in split layer: " << op->get_friendly_name()
                               << " output: " <<  op->get_output_tensor_name(i);
        }
        for (size_t i = 0; i < inputDims.size(); i++) {
            if ((outLayerDims[i] + startOffset[i]) > inputDims[i]) {
                THROW_IE_EXCEPTION << "Invalid dimesions in split layer: " << op->get_friendly_name()
                                   << " output: " <<  op->get_output_tensor_name(i);
            }
        }

        auto outTensor = CldnnTensorFromIEDims(outLayerDims, 1);
        auto offsetTensor = CldnnTensorFromIEDims(startOffset, 0);

        auto cropPrim = cldnn::crop(outLayerName, inputPrimitives[0], outTensor, offsetTensor);
        primitivesToIRLayersMap[outLayerName] = { op->get_friendly_name() };
        primitiveIDs[outLayerName] = outLayerName;

        topology.add(cropPrim);
        profilingIDs.push_back(outLayerName);
        InitProfileInfo(outLayerName, "Crop");

        for (size_t i = 0; i < inputDims.size(); i++) {
            if (outLayerDims[i] != inputDims[i]) {
                startOffset[i] += outLayerDims[i];
            }
        }
    }

    // set split as not_run
    InitProfileInfo(op->get_friendly_name(), op->get_type_name(), false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);
}

void Program::CreateSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Split>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    CreateCommonSplitOp(topology, op);
}

void Program::CreateVariadicSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::VariadicSplit>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {3});
    CreateCommonSplitOp(topology, op);
}

}  // namespace CLDNNPlugin
