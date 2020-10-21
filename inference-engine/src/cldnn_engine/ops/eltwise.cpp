// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "transformations/utils/utils.hpp"

#include "api/eltwise.hpp"
#include "api/reorder.hpp"
#include "api/reshape.hpp"

namespace CLDNNPlugin {

void Program::CreateElementwiseOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op,
                                  cldnn::eltwise_mode mode) {
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outRank = op->get_output_shape(0).size();
    for (size_t i = 0; i < inputPrimitives.size(); ++i) {
        auto inputShape = op->get_input_shape(i);
        auto inputRank = inputShape.size();
        if (inputRank != outRank) {
            // Add reorder if changing number of dimensions requires changing format
            auto targetFormat = DefaultFormatForDims(outRank);
            if (targetFormat.value != DefaultFormatForDims(inputRank).value) {
                auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                auto targetDatatype = DataTypeFromPrecision(op->get_input_element_type(i));
                auto reorderPrim = cldnn::reorder(reorderName, inputPrimitives[i], targetFormat, targetDatatype);

                topology.add(reorderPrim);
                AddInnerPrimitiveToProfiler(reorderName, layerName, op);

                inputPrimitives[i] = reorderName;
            }

            auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

            // Extend input dimensions by prepending ones
            inputShape.insert(inputShape.begin(), outRank - inputRank, 1ul);

            auto targetShape = CldnnTensorFromIEDims(inputShape);

            auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], targetShape);
            topology.add(reshapePrim);
            AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

            inputPrimitives[i] = reshapeName;
        }
    }

    auto out_dt = DataTypeFromPrecision(op->get_output_element_type(0));
    auto eltwisePrim = cldnn::eltwise(layerName,
                                      inputPrimitives,
                                      mode,
                                      {},
                                      out_dt);

    topology.add(eltwisePrim);

    AddPrimitiveToProfiler(op);
}

void Program::CreateAddOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Add>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::sum);
}

void Program::CreateMultiplyOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::prod);
}

void Program::CreateMaximumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Maximum>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::max);
}

void Program::CreateMinimumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Minimum>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::min);
}

void Program::CreateSubtractOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Subtract>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::sub);
}

void Program::CreateDivideOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Divide>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::div);
}

void Program::CreateSquaredDifferenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::SquaredDifference>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::squared_diff);
}

void Program::CreateEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Equal>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::eq);
}

void Program::CreateNotEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::NotEqual>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::ne);
}

void Program::CreateLessOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Less>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::lt);
}

void Program::CreateLessEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::LessEqual>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::le);
}

void Program::CreateGreaterOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Greater>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::gt);
}

void Program::CreateGreaterEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::GreaterEqual>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::ge);
}

void Program::CreateLogicalAndOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::LogicalAnd>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::logic_and);
}

void Program::CreateLogicalOrOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::LogicalOr>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::logic_or);
}

void Program::CreateLogicalXorOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::LogicalXor>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::logic_xor);
}

void Program::CreatePowerOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Power>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::pow);
}

void Program::CreateFloorModOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::FloorMod>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateElementwiseOp(topology, op, cldnn::eltwise_mode::floor_mod);
}

}  // namespace CLDNNPlugin
