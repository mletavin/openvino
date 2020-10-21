// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/tile.hpp"

namespace CLDNNPlugin {

void Program::CreateTileOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Tile>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto tilePrim = cldnn::tile(layerName,
                                inputPrimitives[0],
                                CldnnTensorFromIEDims(op->get_output_shape(0)));

    topology.add(tilePrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
