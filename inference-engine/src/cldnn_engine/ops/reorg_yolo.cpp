// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/reorg_yolo.hpp"

namespace CLDNNPlugin {

void Program::CreateReorgYoloOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::ReorgYolo>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t stride = op->get_strides()[0];

    auto reorgPrim = cldnn::reorg_yolo(layerName,
                                       inputPrimitives[0],
                                       stride);

    topology.add(reorgPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
