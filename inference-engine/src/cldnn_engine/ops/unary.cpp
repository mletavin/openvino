// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "transformations/utils/utils.hpp"

#include "api/activation.hpp"

namespace CLDNNPlugin {

void Program::CreateUnaryEltwiseOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op,
                                 cldnn::activation_func func, cldnn::activation_additional_params params) {
    auto inputs = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto activationPrimitive = cldnn::activation(layerName, inputs[0], func, params);
    topology.add(activationPrimitive);
    AddPrimitiveToProfiler(op);
}

void Program::CreateTanhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Tanh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::hyperbolic_tan, {});
}

void Program::CreateEluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Elu>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    auto alpha = static_cast<float>(op->get_alpha());
    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::elu, {alpha});
}

void Program::CreateSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Sigmoid>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::logistic, {});
}

void Program::CreateReluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Relu>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::relu, {});
}

void Program::CreatePReluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::PRelu>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto slope_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    if (!slope_node) {
        THROW_IE_EXCEPTION << "Unsupported slope node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    if (ngraph::shape_size(slope_node->get_output_shape(0)) == 1) {
        float slope;
        if (!ngraph::op::util::get_single_value(slope_node, slope))
            THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::relu_negative_slope, {slope});
    } else {
        auto inputs = GetInputPrimitiveIDs(op);
        std::string layerName = layer_type_name_ID(op);
        auto activationPrimitive = cldnn::activation(layerName, inputs[0], inputs[1], cldnn::activation_func::relu_negative_slope);
        topology.add(activationPrimitive);
        AddPrimitiveToProfiler(op);
    }
}

void Program::CreateClampOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Clamp>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    float min = static_cast<float>(op->get_min());
    float max = static_cast<float>(op->get_max());
    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::clamp, {min, max});
}

void Program::CreateExpOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Exp>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::exp, {});
}

void Program::CreateNotOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Not>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::negation, {});
}

void Program::CreateLogicalNotOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::LogicalNot>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::negation, {});
}

void Program::CreateAsinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Asin>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::asin, {});
}

void Program::CreateAsinhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v3::Asinh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::asinh, {});
}

void Program::CreateAcosOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Acos>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::acos, {});
}

void Program::CreateAcoshOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v3::Acosh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::acosh, {});
}

void Program::CreateAtanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Atan>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::atan, {});
}

void Program::CreateAtanhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v3::Atanh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::atanh, {});
}

void Program::CreateAbsOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Abs>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::abs, {});
}

void Program::CreateFloorOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Floor>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::floor, {});
}

void Program::CreateCeilingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Ceiling>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::ceil, {});
}

void Program::CreateSqrtOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Sqrt>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::sqrt, {});
}

void Program::CreateErfOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Erf>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::erf, {});
}

void Program::CreateHardSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::HardSigmoid>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto beta_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !beta_node) {
        THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    if (ngraph::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ngraph::shape_size(beta_node->get_output_shape(0)) == 1)  {
        float alpha, beta;
        if (!ngraph::op::util::get_single_value(alpha_node, alpha) || !ngraph::op::util::get_single_value(beta_node, beta)) {
            THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::hard_sigmoid, {alpha, beta});
    }
}

void Program::CreateLogOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Log>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::log, {});
}

void Program::CreateNegativeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Negative>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::negative, {});
}

// void Program::CreateReciprocalOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {

// }

void Program::CreateSeluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Selu>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {3});
    auto alpha_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto lambda_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    if (!alpha_node || !lambda_node) {
        THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    if (ngraph::shape_size(alpha_node->get_output_shape(0)) == 1 &&
        ngraph::shape_size(lambda_node->get_output_shape(0)) == 1)  {
        float alpha, lambda;
        if (!ngraph::op::util::get_single_value(alpha_node, alpha) || !ngraph::op::util::get_single_value(lambda_node, lambda)) {
            THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::selu, {alpha, lambda});
    }
}

void Program::CreateSoftPlusOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::SoftPlus>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::softplus, {});
}

// void Program::CreateSoftSignOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    // auto op = std::dynamic_pointer_cast<ngraph::op::v0::SoftSign>(node);
    // if (!op)
    //     THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;
    // CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::softsign, {});
// }

void Program::CreateTanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Tan>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::tan, {});
}

void Program::CreateSinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Sin>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::sin, {});
}

void Program::CreateSinhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Sinh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::sinh, {});
}

void Program::CreateCosOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Cos>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::cos, {});
}

void Program::CreateCoshOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Cosh>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::cosh, {});
}

void Program::CreateSwishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::Swish>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::swish, {});
}

void Program::CreateHSwishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::HSwish>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::hswish, {});
}

void Program::CreateMishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::Mish>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::mish, {});
}

void Program::CreateGeluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Gelu>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::gelu, {});
}

void Program::CreateSignOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Sign>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::sign, {});
}

void Program::CreateHSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v5::HSigmoid>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    CreateUnaryEltwiseOp(topology, op, cldnn::activation_func::hsigmoid, {});
}

void Program::CreateRoundOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v5::Round>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    auto func = cldnn::activation_func::none;
    switch (op->get_mode()) {
        case ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN : func = cldnn::activation_func::round_half_to_even; break;
        case ngraph::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO : func = cldnn::activation_func::round_half_away_from_zero; break;
        default: THROW_IE_EXCEPTION << "Unsupported round mode in " << op->get_friendly_name() << ": " << static_cast<int>(op->get_mode());
    }
    CreateUnaryEltwiseOp(topology, op, func, {});
}

}  // namespace CLDNNPlugin
