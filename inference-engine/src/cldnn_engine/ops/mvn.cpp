// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/mvn.hpp"

namespace CLDNNPlugin {

void Program::CreateMVNOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::MVN>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    const size_t chanelAxis = 1;
    ngraph::AxisSet reductionAxes = op->get_reduction_axes();
    // FIXME: op->get_across_channels(); doesn't work for some reason. Is it expected?
    bool across_channels = reductionAxes.count(chanelAxis) > 0;
    bool normalize_variance = op->get_normalize_variance();
    float eps = op->get_eps();

    auto mvnPrim = cldnn::mvn(layerName,
                              inputPrimitives[0],
                              across_channels,
                              normalize_variance,
                              eps);

    topology.add(mvnPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
