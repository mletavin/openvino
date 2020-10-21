// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/reorder.hpp"

using namespace InferenceEngine;

namespace CLDNNPlugin {

void Program::CreateResultOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node,
                             InferenceEngine::OutputsDataMap networkOutputs) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Result>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});

    auto prev = op->get_input_node_shared_ptr(0);
    auto inputID = prev->get_friendly_name();
    if (prev->get_output_size() > 1) {
        inputID += "." + std::to_string(op->get_input_source_output(0).get_index());
    }
    auto it = networkOutputs.find(inputID);
    if (it == networkOutputs.end()) {
        THROW_IE_EXCEPTION << "Can't find output " << inputID << " in OutputsDataMap";
    }
    std::string originalOutName = it->first;
    InferenceEngine::DataPtr outputData = it->second;

    auto inputs = GetInputPrimitiveIDs(op);
    const auto outputDesc = outputData->getTensorDesc();
    const auto outputlayout = outputDesc.getLayout();

    // TODO: add precision check once there's an outputInfo object
    if (outputlayout != InferenceEngine::NCHW &&
        // TODO: change 6d case once new layout added in IE
        outputlayout != InferenceEngine::BLOCKED &&
        outputlayout != InferenceEngine::NCDHW &&
        outputlayout != InferenceEngine::NHWC &&
        outputlayout != InferenceEngine::CHW &&
        outputlayout != InferenceEngine::NC &&
        outputlayout != InferenceEngine::C &&
        outputlayout != InferenceEngine::SCALAR) {
        THROW_IE_EXCEPTION << "Unsupported layout (" << outputlayout << ") in output: " << originalOutName;
    }

    auto outLayerName = layer_type_name_ID(op);
    Precision precision = outputData->getPrecision();
    std::string outputID = inputs[0];

    topology.add(cldnn::reorder(outLayerName,
                                outputID,
                                FormatFromLayout(outputData->getLayout()),
                                DataTypeFromPrecision(precision)));
    InitProfileInfo(outLayerName, "reorder");
    profilingIDs.push_back(outLayerName);
    primitiveIDs[outLayerName] = outLayerName;
    primitiveIDs[originalOutName] = outLayerName;

    outputDims[originalOutName] = outputDesc.getDims();
    prevPrimitiveIDs[outLayerName] = {originalOutName};
}

}  // namespace CLDNNPlugin
