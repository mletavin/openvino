// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/gather_tree.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

void Program::CreateGatherTreeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::GatherTree>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {4});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // clDNN primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + m_preProcessTag;
            auto targetFormat = DefaultFormatForDims(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            topology.add(preprocessPrim);
            AddInnerPrimitiveToProfiler(reorderPrimName, layerName, op);
            reorderedInputs[portIndex] = reorderPrimName;
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    auto gatherTreePrim = cldnn::gather_tree(layerName,
                                             reorderedInputs[0],
                                             reorderedInputs[1],
                                             reorderedInputs[2],
                                             reorderedInputs[3]);

    topology.add(gatherTreePrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
