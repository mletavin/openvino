// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/quantize.hpp"

namespace CLDNNPlugin {

void Program::CreateFakeQuantizeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {5});
    std::string layerName = layer_type_name_ID(op);
    auto inputPrimitives = GetInputPrimitiveIDs(op);

    auto input_id       = inputPrimitives[0];
    auto input_low_id   = inputPrimitives[1];
    auto input_high_id  = inputPrimitives[2];
    auto output_low_id  = inputPrimitives[3];
    auto output_high_id = inputPrimitives[4];

    int levels = static_cast<int>(op->get_levels());
    auto dt = DataTypeFromPrecision(op->get_output_element_type(0));
    auto quantizationPrim = cldnn::quantize(layerName,
                                            input_id,
                                            input_low_id,
                                            input_high_id,
                                            output_low_id,
                                            output_high_id,
                                            levels,
                                            dt);

    topology.add(quantizationPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
