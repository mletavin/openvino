// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/convert.hpp"
#include "ngraph/op/convert_like.hpp"

#include "api/reorder.hpp"

namespace CLDNNPlugin {

void Program::CreateConvertLikeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ConvertLike>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = DataTypeFromPrecision(op->get_input_element_type(1));

    auto reorderPrim = cldnn::reorder(layerName, inputPrimitives[0], cldnn::format::any, outDataType);

    topology.add(reorderPrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateConvertOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Convert>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = DataTypeFromPrecision(op->get_destination_type());

    auto reorderPrim = cldnn::reorder(layerName, inputPrimitives[0], cldnn::format::any, outDataType);

    topology.add(reorderPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
