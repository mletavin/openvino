// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/reduce_logical_or.hpp"
#include "ngraph/op/reduce_logical_and.hpp"
#include "ngraph/op/reduce_l1.hpp"
#include "ngraph/op/reduce_l2.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/constant.hpp"

#include "api/reduce.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

void Program::CreateReduceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op,
                             cldnn::reduce_mode mode, bool keep_dims) {
    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t rank = op->get_input_shape(0).size();

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!axes_constant) {
        THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }
    std::vector<int32_t> rawAxes = axes_constant->cast_vector<int32_t>();

    std::vector<uint16_t> axes;
    for (size_t a = 0; a < rawAxes.size(); a++) {
        if (rawAxes[a] < 0)
            rawAxes[a] = rawAxes[a] + rank;
        if (rawAxes[a] < 0 || rawAxes[a] > rank - 1)
            THROW_IE_EXCEPTION << op->get_friendly_name() << " Incorrect Reduce axis value: " << rawAxes[a];
        if (rank == 6) {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_w); break;
                case 3: axes.push_back(cldnn::reduce::along_z); break;
                case 4: axes.push_back(cldnn::reduce::along_y); break;
                case 5: axes.push_back(cldnn::reduce::along_x); break;
            }
        } else if (rank == 5) {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_z); break;
                case 3: axes.push_back(cldnn::reduce::along_y); break;
                case 4: axes.push_back(cldnn::reduce::along_x); break;
            }
        } else {
            switch (rawAxes[a]) {
                case 0: axes.push_back(cldnn::reduce::along_b); break;
                case 1: axes.push_back(cldnn::reduce::along_f); break;
                case 2: axes.push_back(cldnn::reduce::along_y); break;
                case 3: axes.push_back(cldnn::reduce::along_x); break;
            }
        }
    }

    sort(axes.begin(), axes.end());
    axes.erase(unique(axes.begin(), axes.end()), axes.end());

    auto reducePrim = cldnn::reduce(layerName,
                                    inputPrimitives[0],
                                    mode,
                                    axes,
                                    static_cast<int32_t>(keep_dims));

    topology.add(reducePrim);

    auto reorderLayerName = layerName + "_reorder";
    cldnn::format out_format = cldnn::format::any;
    auto out_dt = DataTypeFromPrecision(op->get_output_element_type(0));
    if (!keep_dims && rank > 4) {
        if (rank - rawAxes.size() == 6)
            out_format = cldnn::format::bfwzyx;
        else if (rank - rawAxes.size() == 5)
            out_format = cldnn::format::bfzyx;
        else if (rank - rawAxes.size() <= 4)
            out_format = cldnn::format::bfyx;

        auto reorder_prim = cldnn::reorder(reorderLayerName, layerName, out_format, out_dt);
        topology.add(reorder_prim);
        AddPrimitiveToProfiler(op, reorderLayerName);
    } else {
        AddPrimitiveToProfiler(op);
    }
}

void Program::CreateReduceMaxOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceMax>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::max, op->get_keep_dims());
}

void Program::CreateReduceLogicalAndOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceLogicalAnd>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::logical_and, op->get_keep_dims());
}

void Program::CreateReduceLogicalOrOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceLogicalOr>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::logical_or, op->get_keep_dims());
}

void Program::CreateReduceMeanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceMean>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::mean, op->get_keep_dims());
}

void Program::CreateReduceMinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceMin>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::min, op->get_keep_dims());
}

void Program::CreateReduceProdOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceProd>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::prod, op->get_keep_dims());
}

void Program::CreateReduceSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ReduceSum>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::sum, op->get_keep_dims());
}

void Program::CreateReduceL1Op(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::ReduceL1>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::l1, op->get_keep_dims());
}

void Program::CreateReduceL2Op(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::ReduceL2>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateReduceOp(topology, op, cldnn::reduce_mode::l2, op->get_keep_dims());
}

}  // namespace CLDNNPlugin
