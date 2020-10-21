// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/ctc_greedy_decoder.hpp"

namespace CLDNNPlugin {

void Program::CreateCTCGreedyDecoderOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::CTCGreedyDecoder>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::ctc_greedy_decoder(layerName,
                                               inputPrimitives[0],
                                               inputPrimitives[1],
                                               op->get_ctc_merge_repeated(),
                                               DataTypeFromPrecision(op->get_output_element_type(0)),
                                               CldnnTensorFromIEDims(op->get_output_shape(0)));

    topology.add(primitive);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
