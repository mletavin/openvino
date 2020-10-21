// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/reverse_sequence.hpp"

namespace CLDNNPlugin {

void Program::CreateReverseSequenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::ReverseSequence>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t batch_axis = op->get_batch_axis();
    size_t seq_axis = op->get_sequence_axis();
    auto reverseSequencePrim = cldnn::reverse_sequence(layerName,
                                                       inputPrimitives[0],
                                                       inputPrimitives[1],
                                                       seq_axis,
                                                       batch_axis);

    topology.add(reverseSequencePrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
