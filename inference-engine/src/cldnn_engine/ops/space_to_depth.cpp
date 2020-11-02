// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/space_to_depth.hpp"

#include "api/space_to_depth.hpp"

namespace CLDNNPlugin {

static cldnn::space_to_depth::depth_mode GetDepthMode(ngraph::op::v0::SpaceToDepth::SpaceToDepthMode mode) {
    switch (mode) {
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST: return cldnn::space_to_depth::blocks_first;
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST: return cldnn::space_to_depth::depth_first;
        default: THROW_IE_EXCEPTION << "Unsupported SpaceToDepthMode value: " << static_cast<int>(mode);
    }
    return cldnn::space_to_depth::blocks_first;
}

void Program::CreateSpaceToDepthOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::SpaceToDepth>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto spaceToDepthPrim = cldnn::space_to_depth(layerName,
                                                  inputPrimitives[0],
                                                  GetDepthMode(op->get_mode()),
                                                  op->get_block_size());

    topology.add(spaceToDepthPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
