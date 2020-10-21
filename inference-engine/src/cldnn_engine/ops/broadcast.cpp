// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/broadcast.hpp"

namespace CLDNNPlugin {

void Program::CreateCommonBroadcastOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op) {
    ValidateInputs(op, {2, 3});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto broadcastPrim = cldnn::broadcast(layerName,
                                          inputPrimitives[0],
                                          CldnnTensorFromIEDims(op->get_output_shape(0)));

    topology.add(broadcastPrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateBroadcastOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    if (std::dynamic_pointer_cast<ngraph::op::v3::Broadcast>(node) == nullptr &&
        std::dynamic_pointer_cast<ngraph::op::v1::Broadcast>(node) == nullptr)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateCommonBroadcastOp(topology, node);
}

}  // namespace CLDNNPlugin
