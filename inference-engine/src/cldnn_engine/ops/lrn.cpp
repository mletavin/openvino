// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/lrn.hpp"

namespace CLDNNPlugin {

static cldnn::lrn_norm_region GetNormRegion(std::vector<int64_t> axis_value) {
    if (axis_value.size() == 1 && axis_value[0] == 1) {
        return cldnn::lrn_norm_region_across_channel;
    } else {
        return cldnn::lrn_norm_region_within_channel;
    }
}

void Program::CreateLRNOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::LRN>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto axis_const = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (!axis_const) {
        THROW_IE_EXCEPTION << "Unsupported axes node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    auto axis_value = axis_const->cast_vector<int64_t>();
    auto localSize = op->get_nsize();

    auto lrnPrim = cldnn::lrn(layerName,
                              inputPrimitives[0],
                              localSize,
                              static_cast<float>(op->get_bias()),
                              static_cast<float>(op->get_alpha()),
                              static_cast<float>(op->get_beta()),
                              GetNormRegion(axis_value));

    topology.add(lrnPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin