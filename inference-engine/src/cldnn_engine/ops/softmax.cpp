// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/softmax.hpp"

namespace CLDNNPlugin {

static cldnn::softmax::dimension_t toSoftmaxAxis(size_t axis, size_t rank) {
    switch (axis) {
    // FIXME: it seems that axis=0 should correspond to normalize_b;
    case 0: return cldnn::softmax::normalize_all;
    case 1: return cldnn::softmax::normalize_f;
    case 2:
        if (rank > 4)
            return cldnn::softmax::normalize_z;
        else
            return cldnn::softmax::normalize_y;
    case 3:
        if (rank > 4)
            return cldnn::softmax::normalize_y;
        else
            return cldnn::softmax::normalize_x;
    case 4:
        return cldnn::softmax::normalize_x;
    default: THROW_IE_EXCEPTION << "Invalid softmax axis " << axis;
    }
    return cldnn::softmax::normalize_fyx;
}

void Program::CreateSoftmaxOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Softmax>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputPrimitives[0],
                                      toSoftmaxAxis(op->get_axis(), op->get_input_shape(0).size()));
    topology.add(softmaxPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
