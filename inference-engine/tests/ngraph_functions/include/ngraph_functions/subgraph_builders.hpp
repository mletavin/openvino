// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_precision.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
static std::shared_ptr<ngraph::Function> makeConvPoolRelu(std::vector<size_t> inputShape = {1, 1, 32, 32},
                                                      InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params.front(),
                                                          ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4},
                                                                                           ngraph::Shape{1, 32, 1, 32}), false);
    auto conv1 = ngraph::builder::makeConvolution(reshape1, ngPrc, {1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 4);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<size_t> stride{1, 1}, padB{0, 0}, padE = padB, kernel{1, 2};
    auto pool1 = std::make_shared<ngraph::opset1::MaxPool>(conv1, stride, padB, padE, kernel,
                                                               ngraph::op::RoundingType::FLOOR,
                                                               ngraph::op::PadType::EXPLICIT);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(pool1);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(relu1,
                                                              ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                                                                                               ngraph::Shape{1, 116}), false);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    return fnPtr;
}

static std::shared_ptr<ngraph::Function> makeSplitConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20},
                                                            InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    return fnPtr;
}

static std::shared_ptr<ngraph::Function> makeSplitMultiConvConcat(std::vector<size_t> inputShape = {1, 4, 20, 20}) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1_0 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_0 = std::make_shared<ngraph::opset1::Relu>(conv1_0);
    auto conv1_1 = ngraph::builder::makeConvolution(relu1_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_1 = std::make_shared<ngraph::opset1::Relu>(conv1_1);
    auto conv1_2 = ngraph::builder::makeConvolution(relu1_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_2 = std::make_shared<ngraph::opset1::Relu>(conv1_2);
    auto conv1_3 = ngraph::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_3 = std::make_shared<ngraph::opset1::Relu>(conv1_3);
    auto conv1_4 = ngraph::builder::makeConvolution(relu1_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu1_4 = std::make_shared<ngraph::opset1::Relu>(conv1_4);

    auto conv2_0 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_0 = std::make_shared<ngraph::opset1::Relu>(conv2_0);
    auto conv2_1 = ngraph::builder::makeConvolution(relu2_0, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_1 = std::make_shared<ngraph::opset1::Relu>(conv2_1);
    auto conv2_2 = ngraph::builder::makeConvolution(relu2_1, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_2 = std::make_shared<ngraph::opset1::Relu>(conv2_2);
    auto conv2_3 = ngraph::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_3 = std::make_shared<ngraph::opset1::Relu>(conv2_3);
    auto conv2_4 = ngraph::builder::makeConvolution(relu2_2, ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                    ngraph::op::PadType::EXPLICIT, 5);
    auto relu2_4 = std::make_shared<ngraph::opset1::Relu>(conv2_4);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1_4->output(0), relu2_4->output(0)}, 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(results, params);
    return fnPtr;
}

static std::shared_ptr<ngraph::Function>
makeTIwithLSTMcell(InferenceEngine::Precision prc = InferenceEngine::Precision::FP32) {
    auto ngPRC = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
    // That which we iterate over
    const size_t N = 32; // Batch size
    const size_t L = 10; // Sequence length
    const size_t I = 8;  // Input size
    const size_t H = 32; // Hidden size
    auto SENT = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, L, I});

    auto H_init = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});
    auto C_init = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});

    auto H_t = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});
    auto C_t = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, H});

    // Body
    auto X = std::make_shared<ngraph::opset1::Parameter>(ngPRC, ngraph::Shape{N, 1, I});
    std::vector<uint64_t> dataW(4 * H * I, 0);
    auto W_body = std::make_shared<ngraph::opset1::Constant>(ngPRC, ngraph::Shape{4 * H, I}, dataW);
    std::vector<uint64_t> dataR(4 * H * H, 0);
    auto R_body = std::make_shared<ngraph::opset1::Constant>(ngPRC, ngraph::Shape{4 * H, H}, dataR);
    std::vector<uint64_t> inShape = {N, H};
    auto constantH = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, inShape);
    inShape = {N, I};
    auto constantX = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, inShape);
    auto LSTM_cell =
            std::make_shared<ngraph::opset1::LSTMCell>(std::make_shared<ngraph::opset1::Reshape>(X, constantX, false),
                                                   std::make_shared<ngraph::opset1::Reshape>(H_t, constantH, false),
                                                   std::make_shared<ngraph::opset1::Reshape>(C_t, constantH, false),
                                                   W_body,
                                                   R_body,
                                                   H);
    inShape = {N, 1, H};
    auto constantHo = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{3}, inShape);
    auto H_o = std::make_shared<ngraph::opset1::Reshape>(LSTM_cell->output(0), constantHo, false);
    auto C_o = std::make_shared<ngraph::opset1::Reshape>(LSTM_cell->output(1), constantHo, false);
    auto body = std::make_shared<ngraph::op::TensorIterator::BodyLambda>(
            ngraph::OutputVector{H_o, C_o}, ngraph::ParameterVector{X, H_t, C_t});

    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
    tensor_iterator->set_body(body);
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, 1);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(out0),
                                        std::make_shared<ngraph::opset1::Result>(out1)};
    auto fn_ptr = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{SENT, H_init, C_init});
    return fn_ptr;
}

static std::shared_ptr<ngraph::Function> makeSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                        InferenceEngine::Precision prc = InferenceEngine::Precision::FP32) {
    ngraph::element::Type type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));

    auto conv1 = ngraph::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto result = std::make_shared<ngraph::opset1::Result>(conv1);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
    return
            fn_ptr;
}

static std::shared_ptr<ngraph::Function> makeMultiSingleConv(std::vector<size_t> inputShape = {1, 3, 24, 24}) {
    ngraph::element::Type type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto conv1 = ngraph::builder::makeConvolution(param0, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv2 = ngraph::builder::makeConvolution(conv1, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv3 = ngraph::builder::makeConvolution(conv2, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv4 = ngraph::builder::makeConvolution(conv3, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv5 = ngraph::builder::makeConvolution(conv4, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv6 = ngraph::builder::makeConvolution(conv5, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv7 = ngraph::builder::makeConvolution(conv6, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv8 = ngraph::builder::makeConvolution(conv7, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                 ngraph::op::PadType::EXPLICIT, 5);
    auto conv9 = ngraph::builder::makeConvolution(conv8, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto conv10 = ngraph::builder::makeConvolution(conv9, type, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto result = std::make_shared<ngraph::opset1::Result>(conv10);
    auto fn_ptr = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
    return
            fn_ptr;
}

static std::shared_ptr<ngraph::Function> make2InputSubtract(std::vector<size_t> inputShape = {1, 3, 24, 24},
                                                            InferenceEngine::Precision prc = InferenceEngine::Precision::FP32) {
    ngraph::element::Type type = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
    auto param0 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(inputShape));
    auto subtract = std::make_shared<ngraph::opset1::Subtract>(param0, param1);
    auto result = std::make_shared<ngraph::opset1::Result>(subtract);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0, param1});
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph