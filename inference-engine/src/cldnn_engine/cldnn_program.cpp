// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "ngraph/ops.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

const cldnn::primitive_id Program::m_preProcessTag("_cldnn_input_preprocess");
const cldnn::primitive_id Program::m_meanValuesTag("_cldnn_mean_values");
const cldnn::primitive_id Program::m_preCustomLayerTag("_cldnn_custom_preprocess");
const cldnn::primitive_id Program::m_postCustomLayerTag("_cldnn_custom_postprocess");

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

std::string layer_type_lower(const ngraph::Node* op) {
    std::string layerType = op->get_type_name();
    std::transform(layerType.begin(), layerType.end(), layerType.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return layerType;
}

std::string layer_type_name_ID(const ngraph::Node* op) {
    return layer_type_lower(op) + ":" + op->get_friendly_name();
}

std::string layer_type_lower(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_lower(op.get());
}

std::string layer_type_name_ID(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_name_ID(op.get());
}

void Program::changeInputBatch(int batch) {
    m_curBatch = batch;
}

void Program::ValidateInputs(const std::shared_ptr<ngraph::Node>& op, std::vector<size_t> validInputsCount) {
    for (auto ic : validInputsCount) {
        if (op->get_input_size() == ic) {
            return;
        }
    }

    THROW_IE_EXCEPTION << "Invalid inputs count (" << op->get_input_size() << ") in "
                       << op->get_friendly_name() << " (" << op->get_type_name()
                       << " op::v" << op->get_type_info().version << ")";
}

bool Program::CanProcessDynBatch(std::vector<std::shared_ptr<ngraph::Node>> ops, InferenceEngine::InputsDataMap networkInputs) const {
    if (networkInputs.empty())
        return false;

    for (auto op : ops) {
        // TODO: do we have any other exception cases?
        if (std::dynamic_pointer_cast<ngraph::op::v0::Reshape>(op)) {
            if (op->get_input_shape(0)[0] == op->get_output_shape(0)[0])
                continue;
        }

        // List of the operations which can lead to invalid dynamic batch processing
        if (std::dynamic_pointer_cast<ngraph::op::v4::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v3::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v1::NonMaxSuppression>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::PSROIPooling>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::ROIPooling>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::PriorBox>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::DetectionOutput>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Reshape>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v4::Proposal>(op) ||
            std::dynamic_pointer_cast<ngraph::op::v0::Proposal>(op)) {
            return false;
        }

        auto customLayer = m_config.customLayers.find(op->get_type_name());
        if (customLayer != m_config.customLayers.end()) {
            return false;
        }
    }

    return true;
}

Program::Program(InferenceEngine::ICNNNetwork& network, std::shared_ptr<const cldnn::engine> engine, const Config& config)
    : m_config(config)
    , m_engine(engine)
    , m_curBatch(-1)
    , queryMode(false) {
    // Extract inputs/outputs info from CNNNetwork
    InferenceEngine::InputsDataMap networkInputs;
    network.getInputsInfo(networkInputs);

    InferenceEngine::OutputsDataMap networkOutputs;
    network.getOutputsInfo(networkOutputs);

    if (networkInputs.empty()) {
        THROW_IE_EXCEPTION << "No inputs detected.";
    }

    auto func = network.getFunction();
    if (!func) {
        THROW_IE_EXCEPTION << "Function pointer inside CNNNetwork is nullptr";
    }

    auto ops = func->get_ordered_ops();

    if (m_config.max_dynamic_batch > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(ops, networkInputs)) {
            THROW_IE_EXCEPTION << "Such topology cannot be compiled for dynamic batch!";
        }
    }

    int m_bv_sz = GetMaxBatchSizeForSingleProgram();

    m_max_batch = config.max_dynamic_batch;

    if (config.max_dynamic_batch > 1) {
        for (int b = m_bv_sz - 1; b >= 0; b--) {
            inputLayouts.clear();
            outputDims.clear();
            primitiveIDs.clear();
            blobMemCache.clear();

            changeInputBatch(1U << static_cast<unsigned>(b));
            m_programs.insert(m_programs.begin(), BuildProgram(ops, networkInputs, networkOutputs));
            m_engine->release_pending_memory(0);
        }
    } else {
        m_programs.emplace_back(BuildProgram(ops, networkInputs, networkOutputs));
        m_engine->release_pending_memory(0);
    }
}

int Program::GetMaxBatchSizeForSingleProgram() {
    if (m_config.max_dynamic_batch > 1) {
        // calculate number of networks necessary based on binary log
        unsigned int tmp = m_config.max_dynamic_batch;
        unsigned int mask = 1U << 31;
        unsigned int ldigit = 31;

        while (!(tmp & mask)) {
            mask >>= 1;
            ldigit--;
        }

        return ldigit + 1;
    }

    return 0;
}

std::shared_ptr<cldnn::program> Program::getCompiledProgram(int program_id) {
    if (program_id >= m_programs.size())
        THROW_IE_EXCEPTION << "Invalid program ID";

    return m_programs[program_id];
}

std::shared_ptr<cldnn::program> Program::BuildProgram(std::vector<std::shared_ptr<ngraph::Node>> ops,
                                                      InferenceEngine::InputsDataMap networkInputs,
                                                      InferenceEngine::OutputsDataMap networkOutputs) {
    cldnn::build_options options;
    if (!m_config.graph_dumps_dir.empty()) {
        options.set_option(cldnn::build_option::graph_dumps_dir(m_config.graph_dumps_dir));
    }
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::tuning_config(m_config.tuningConfig));

    cldnn::topology topology;

    for (auto op : ops) {
        CreateSingleLayerPrimitive(topology, op, networkInputs, networkOutputs);
    }

    return std::make_shared<cldnn::program>(*m_engine, topology, options);
}

bool Program::IsOpSupported(const InferenceEngine::ICNNNetwork& network, const std::shared_ptr<ngraph::Node>& op) {
    InferenceEngine::InputsDataMap networkInputs;
    network.getInputsInfo(networkInputs);

    InferenceEngine::OutputsDataMap networkOutputs;
    network.getOutputsInfo(networkOutputs);

    cldnn::topology topology;
    try {
        // Query mode disables checks that input primitives are created,
        // as IsOpSupported method is called for each operation separately
        // So we just ensure that inputs count is valid for given operation
        EnableQueryMode();
        // Creating topology object for each operation is supposed to be more time-consuming than
        // simple check by op type, but it has 2 big advantages:
        // 1. Code reuse. We don't need to have separate white-list of supported operations or
        //    add any ugly macro/templates to apply single dunction to multiple cases.
        // 2. We also check parameters of each operation, which means we have more
        //    reliable results of QueryNetwork call.
        CreateSingleLayerPrimitive(topology, op, networkInputs, networkOutputs);
        DisableQueryMode();
    } catch (std::exception& ex) {
        // Exception means that an operation or some of it's parameters are not supported
        return false;
    }

    return true;
}

void Program::CreateSingleLayerPrimitive(cldnn::topology& topology,
                                         const std::shared_ptr<ngraph::Node>& op,
                                         InputsDataMap networkInputs,
                                         OutputsDataMap networkOutputs) {
    std::map<ngraph::NodeTypeInfo, std::function<void()>> factories = {
    { ngraph::op::v0::Parameter::type_info, [&](){ CreateParameterOp(topology, op, networkInputs); } },
    { ngraph::op::v0::Result::type_info, [&](){ CreateResultOp(topology, op, networkOutputs); } },
    { ngraph::op::v0::Constant::type_info, [&](){ CreateConstantOp(topology, op); } },
    { ngraph::op::v0::Tanh::type_info, [&](){ CreateTanhOp(topology, op); } },
    { ngraph::op::v0::Elu::type_info, [&](){ CreateEluOp(topology, op); } },
    { ngraph::op::v0::Sigmoid::type_info, [&](){ CreateSigmoidOp(topology, op); } },
    { ngraph::op::v0::Relu::type_info, [&](){ CreateReluOp(topology, op); } },
    { ngraph::op::v0::Clamp::type_info, [&](){ CreateClampOp(topology, op); } },
    { ngraph::op::v0::Exp::type_info, [&](){ CreateExpOp(topology, op); } },
    { ngraph::op::v0::Asin::type_info, [&](){ CreateAsinOp(topology, op); } },
    { ngraph::op::v0::Acos::type_info, [&](){ CreateAcosOp(topology, op); } },
    { ngraph::op::v0::Atan::type_info, [&](){ CreateAtanOp(topology, op); } },
    { ngraph::op::v0::Abs::type_info, [&](){ CreateAbsOp(topology, op); } },
    { ngraph::op::v0::Floor::type_info, [&](){ CreateFloorOp(topology, op); } },
    { ngraph::op::v0::Ceiling::type_info, [&](){ CreateCeilingOp(topology, op); } },
    { ngraph::op::v0::Sqrt::type_info, [&](){ CreateSqrtOp(topology, op); } },
    { ngraph::op::v0::Erf::type_info, [&](){ CreateErfOp(topology, op); } },
    { ngraph::op::v0::HardSigmoid::type_info, [&](){ CreateHardSigmoidOp(topology, op); } },
    { ngraph::op::v0::Log::type_info, [&](){ CreateLogOp(topology, op); } },
    { ngraph::op::v0::Negative::type_info, [&](){ CreateNegativeOp(topology, op); } },
    { ngraph::op::v0::Selu::type_info, [&](){ CreateSeluOp(topology, op); } },
    { ngraph::op::v0::Tan::type_info, [&](){ CreateTanOp(topology, op); } },
    { ngraph::op::v0::Sin::type_info, [&](){ CreateSinOp(topology, op); } },
    { ngraph::op::v0::Sinh::type_info, [&](){ CreateSinhOp(topology, op); } },
    { ngraph::op::v0::Cos::type_info, [&](){ CreateCosOp(topology, op); } },
    { ngraph::op::v0::Cosh::type_info, [&](){ CreateCoshOp(topology, op); } },
    { ngraph::op::v0::Gelu::type_info, [&](){ CreateGeluOp(topology, op); } },
    { ngraph::op::v0::Sign::type_info, [&](){ CreateSignOp(topology, op); } },
    { ngraph::op::v0::SquaredDifference::type_info, [&](){ CreateSquaredDifferenceOp(topology, op); } },
    { ngraph::op::v0::PRelu::type_info, [&](){ CreatePReluOp(topology, op); } },
    { ngraph::op::v0::SpaceToDepth::type_info, [&](){ CreateSpaceToDepthOp(topology, op); } },
    { ngraph::op::v0::DepthToSpace::type_info, [&](){ CreateDepthToSpaceOp(topology, op); } },
    { ngraph::op::v0::CumSum::type_info, [&](){ CreateCumSumOp(topology, op); } },
    { ngraph::op::v0::FakeQuantize::type_info, [&](){ CreateFakeQuantizeOp(topology, op); } },
    { ngraph::op::v0::Squeeze::type_info, [&](){ CreateSqueezeOp(topology, op); } },
    { ngraph::op::v0::Unsqueeze::type_info, [&](){ CreateUnsqueezeOp(topology, op); } },
    { ngraph::op::v0::PriorBox::type_info, [&](){ CreatePriorBoxOp(topology, op); } },
    { ngraph::op::v0::PriorBoxClustered::type_info, [&](){ CreatePriorBoxClusteredOp(topology, op); } },
    { ngraph::op::v0::MatMul::type_info, [&](){ CreateMatMulOp(topology, op); } },
    { ngraph::op::v0::ShuffleChannels::type_info, [&](){ CreateShuffleChannelsOp(topology, op); } },
    { ngraph::op::v0::DetectionOutput::type_info, [&](){ CreateDetectionOutputOp(topology, op); } },
    { ngraph::op::v0::Concat::type_info, [&](){ CreateConcatOp(topology, op); } },
    { ngraph::op::v0::ROIPooling::type_info, [&](){ CreateROIPoolingOp(topology, op); } },
    { ngraph::op::v0::PSROIPooling::type_info, [&](){ CreatePSROIPoolingOp(topology, op); } },
    { ngraph::op::v0::NormalizeL2::type_info, [&](){ CreateNormalizeL2Op(topology, op); } },
    { ngraph::op::v0::MVN::type_info, [&](){ CreateMVNOp(topology, op); } },
    { ngraph::op::v0::GRN::type_info, [&](){ CreateGRNOp(topology, op); } },
    { ngraph::op::v0::LRN::type_info, [&](){ CreateLRNOp(topology, op); } },
    { ngraph::op::v0::Tile::type_info, [&](){ CreateTileOp(topology, op); } },
    { ngraph::op::v0::Convert::type_info, [&](){ CreateConvertOp(topology, op); } },
    { ngraph::op::v0::CTCGreedyDecoder::type_info, [&](){ CreateCTCGreedyDecoderOp(topology, op); } },
    { ngraph::op::v0::RegionYolo::type_info, [&](){ CreateRegionYoloOp(topology, op); } },
    { ngraph::op::v0::ReorgYolo::type_info, [&](){ CreateReorgYoloOp(topology, op); } },
    { ngraph::op::v0::Interpolate::type_info, [&](){ CreateInterpolateOp(topology, op); } },
    { ngraph::op::v0::ReverseSequence::type_info, [&](){ CreateReverseSequenceOp(topology, op); } },
    { ngraph::op::v0::Proposal::type_info, [&](){ CreateProposalOp(topology, op); } },
    // { ngraph::op::v0::LSTMSequence::type_info, [&](){ CreateLSTMSequenceOp(topology, op); } },
    // { ngraph::op::v0::LSTMCell::type_info, [&](){ CreateLSTMCellOp(topology, op); } },
    // { ngraph::op::v0::ShapeOf::type_info, [&](){ CreateShapeOfOp(topology, op); } },
    // { ngraph::op::v0::Range::type_info, [&](){ CreateRangeOp(topology, op); } },
    // { ngraph::op::v0::BatchNormInference::type_info, [&](){ CreateBatchNormInferenceOp(topology, op); } },
    // { ngraph::op::v0::RNNCell::type_info, [&](){ CreateRNNCellOp(topology, op); } },
    // { ngraph::op::v0::TensorIterator::type_info, [&](){ CreateTensorIteratorOp(topology, op); } },

    { ngraph::op::v1::ReduceMax::type_info, [&](){ CreateReduceMaxOp(topology, op); } },
    { ngraph::op::v1::ReduceLogicalAnd::type_info, [&](){ CreateReduceLogicalAndOp(topology, op); } },
    { ngraph::op::v1::ReduceLogicalOr::type_info, [&](){ CreateReduceLogicalOrOp(topology, op); } },
    { ngraph::op::v1::ReduceMean::type_info, [&](){ CreateReduceMeanOp(topology, op); } },
    { ngraph::op::v1::ReduceMin::type_info, [&](){ CreateReduceMinOp(topology, op); } },
    { ngraph::op::v1::ReduceProd::type_info, [&](){ CreateReduceProdOp(topology, op); } },
    { ngraph::op::v1::ReduceSum::type_info, [&](){ CreateReduceSumOp(topology, op); } },
    { ngraph::op::v1::Add::type_info, [&](){ CreateAddOp(topology, op); } },
    { ngraph::op::v1::Subtract::type_info, [&](){ CreateSubtractOp(topology, op); } },
    { ngraph::op::v1::Divide::type_info, [&](){ CreateDivideOp(topology, op); } },
    { ngraph::op::v1::Multiply::type_info, [&](){ CreateMultiplyOp(topology, op); } },
    { ngraph::op::v1::Maximum::type_info, [&](){ CreateMaximumOp(topology, op); } },
    { ngraph::op::v1::Minimum::type_info, [&](){ CreateMinimumOp(topology, op); } },
    { ngraph::op::v1::Equal::type_info, [&](){ CreateEqualOp(topology, op); } },
    { ngraph::op::v1::NotEqual::type_info, [&](){ CreateNotEqualOp(topology, op); } },
    { ngraph::op::v1::Less::type_info, [&](){ CreateLessOp(topology, op); } },
    { ngraph::op::v1::LessEqual::type_info, [&](){ CreateLessEqualOp(topology, op); } },
    { ngraph::op::v1::Greater::type_info, [&](){ CreateGreaterOp(topology, op); } },
    { ngraph::op::v1::GreaterEqual::type_info, [&](){ CreateGreaterEqualOp(topology, op); } },
    { ngraph::op::v1::LogicalNot::type_info, [&](){ CreateLogicalNotOp(topology, op); } },
    { ngraph::op::v1::LogicalAnd::type_info, [&](){ CreateLogicalAndOp(topology, op); } },
    { ngraph::op::v1::LogicalOr::type_info, [&](){ CreateLogicalOrOp(topology, op); } },
    { ngraph::op::v1::LogicalXor::type_info, [&](){ CreateLogicalXorOp(topology, op); } },
    { ngraph::op::v1::Power::type_info, [&](){ CreatePowerOp(topology, op); } },
    { ngraph::op::v1::FloorMod::type_info, [&](){ CreateFloorModOp(topology, op); } },
    { ngraph::op::v1::Convolution::type_info, [&](){ CreateConvolutionOp(topology, op); } },
    { ngraph::op::v1::GroupConvolution::type_info, [&](){ CreateGroupConvolutionOp(topology, op); } },
    { ngraph::op::v1::ConvolutionBackpropData::type_info, [&](){ CreateConvolutionBackpropDataOp(topology, op); } },
    { ngraph::op::v1::GroupConvolutionBackpropData::type_info, [&](){ CreateGroupConvolutionBackpropDataOp(topology, op); } },
    { ngraph::op::v1::DeformableConvolution::type_info, [&](){ CreateDeformableConvolutionOp(topology, op); } },
    { ngraph::op::v1::BinaryConvolution::type_info, [&](){ CreateBinaryConvolutionOp(topology, op); } },
    { ngraph::op::v1::MaxPool::type_info, [&](){ CreateMaxPoolOp(topology, op); } },
    { ngraph::op::v1::AvgPool::type_info, [&](){ CreateAvgPoolOp(topology, op); } },
    { ngraph::op::v1::BatchToSpace::type_info, [&](){ CreateBatchToSpaceOp(topology, op); } },
    { ngraph::op::v1::SpaceToBatch::type_info, [&](){ CreateSpaceToBatchOp(topology, op); } },
    { ngraph::op::v1::Softmax::type_info, [&](){ CreateSoftmaxOp(topology, op); } },
    { ngraph::op::v1::Gather::type_info, [&](){ CreateGatherOp(topology, op); } },
    { ngraph::op::v1::GatherTree::type_info, [&](){ CreateGatherTreeOp(topology, op); } },
    { ngraph::op::v1::Reshape::type_info, [&](){ CreateReshapeOp(topology, op); } },
    { ngraph::op::v1::Transpose::type_info, [&](){ CreateTransposeOp(topology, op); } },
    { ngraph::op::v1::TopK::type_info, [&](){ CreateTopKOp(topology, op); } },
    { ngraph::op::v1::DeformablePSROIPooling::type_info, [&](){ CreateDeformablePSROIPoolingOp(topology, op); } },
    { ngraph::op::v1::StridedSlice::type_info, [&](){ CreateStridedSliceOp(topology, op); } },
    { ngraph::op::v1::Pad::type_info, [&](){ CreatePadOp(topology, op); } },
    { ngraph::op::v1::Broadcast::type_info, [&](){ CreateBroadcastOp(topology, op); } },
    { ngraph::op::v1::OneHot::type_info, [&](){ CreateOneHotOp(topology, op); } },
    { ngraph::op::v1::ConvertLike::type_info, [&](){ CreateConvertLikeOp(topology, op); } },
    { ngraph::op::v1::Select::type_info, [&](){ CreateSelectOp(topology, op); } },
    { ngraph::op::v1::Split::type_info, [&](){ CreateSplitOp(topology, op); } },
    { ngraph::op::v1::VariadicSplit::type_info, [&](){ CreateVariadicSplitOp(topology, op); } },
    // { ngraph::op::v1::NonMaxSuppression::type_info, [&](){ CreateNonMaxSuppressionOp(topology, op); } },
    // { ngraph::op::v1::Reverse::type_info, [&](){ CreateReverseOp(topology, op); } },

    { ngraph::op::v3::Asinh::type_info, [&](){ CreateAsinhOp(topology, op); } },
    { ngraph::op::v3::Acosh::type_info, [&](){ CreateAcoshOp(topology, op); } },
    { ngraph::op::v3::Atanh::type_info, [&](){ CreateAtanhOp(topology, op); } },
    { ngraph::op::v3::ExtractImagePatches::type_info, [&](){ CreateExtractImagePatchesOp(topology, op); } },
    { ngraph::op::v3::EmbeddingBagOffsetsSum::type_info, [&](){ CreateEmbeddingBagOffsetsSumOp(topology, op); } },
    { ngraph::op::v3::EmbeddingBagPackedSum::type_info, [&](){ CreateEmbeddingBagPackedSumOp(topology, op); } },
    { ngraph::op::v3::EmbeddingSegmentsSum::type_info, [&](){ CreateEmbeddingSegmentsSumOp(topology, op); } },
    { ngraph::op::v3::Broadcast::type_info, [&](){ CreateBroadcastOp(topology, op); } },
    { ngraph::op::v3::ScatterUpdate::type_info, [&](){ CreateScatterUpdateOp(topology, op); } },
    // { ngraph::op::v3::TopK::type_info, [&](){ CreateTopKOp(topology, op); } },
    // { ngraph::op::v3::ScatterElementsUpdate::type_info, [&](){ CreateScatterElementsUpdateOp(topology, op); } },
    // { ngraph::op::v3::ROIAlign::type_info, [&](){ CreateROIAlignOp(topology, op); } },
    // { ngraph::op::v3::ScatterNDUpdate::type_info, [&](){ CreateScatterNDUpdateOp(topology, op); } },
    // { ngraph::op::v3::Bucketize::type_info, [&](){ CreateBucketizeOp(topology, op); } },
    // { ngraph::op::v3::GRUCell::type_info, [&](){ CreateGRUCellOp(topology, op); } },
    // { ngraph::op::v3::ShapeOf::type_info, [&](){ CreateShapeOfOp(topology, op); } },
    // { ngraph::op::v3::NonZero::type_info, [&](){ CreateNonZeroOp(topology, op); } },
    // { ngraph::op::v3::Assign::type_info, [&](){ CreateAssignOp(topology, op); } },
    // { ngraph::op::v3::ReadValue::type_info, [&](){ CreateReadValueOp(topology, op); } },

    { ngraph::op::v4::SoftPlus::type_info, [&](){ CreateSoftPlusOp(topology, op); } },
    { ngraph::op::v4::Swish::type_info, [&](){ CreateSwishOp(topology, op); } },
    { ngraph::op::v4::HSwish::type_info, [&](){ CreateHSwishOp(topology, op); } },
    { ngraph::op::v4::Mish::type_info, [&](){ CreateMishOp(topology, op); } },
    { ngraph::op::v4::ReduceL1::type_info, [&](){ CreateReduceL1Op(topology, op); } },
    { ngraph::op::v4::ReduceL2::type_info, [&](){ CreateReduceL2Op(topology, op); } },
    { ngraph::op::v4::Proposal::type_info, [&](){ CreateProposalOp(topology, op); } },
    { ngraph::op::v4::NonMaxSuppression::type_info, [&](){ CreateNonMaxSuppressionOp(topology, op); } },
    { ngraph::op::v4::Interpolate::type_info, [&](){ CreateInterpolateOp(topology, op); } },
    { ngraph::op::v4::LSTMCell::type_info, [&](){ CreateLSTMCellOp(topology, op); } },
    // { ngraph::op::v4::CTCLoss::type_info, [&](){ CreateCTCLossOp(topology, op); } },
    // { ngraph::op::v4::Range::type_info, [&](){ CreateRangeOp(topology, op); } },

    { ngraph::op::v5::LSTMSequence::type_info, [&](){ CreateLSTMSequenceOp(topology, op); } },
    { ngraph::op::v5::HSigmoid::type_info, [&](){ CreateHSigmoidOp(topology, op); } },
    { ngraph::op::v5::Round::type_info, [&](){ CreateRoundOp(topology, op); } },
    };

    InitProfileInfo(op->get_friendly_name(), op->get_type_name());

    bool is_created = false;
    const ngraph::NodeTypeInfo* op_type_info = &op->get_type_info();
    while (op_type_info != nullptr) {
        auto customLayer = m_config.customLayers.find(op->get_type_name());
        if (customLayer != m_config.customLayers.end()) {
            CreateCustomOp(topology, op, customLayer->second);
            return;
        }

        auto factory_it = factories.find(*op_type_info);
        if (factory_it != factories.end()) {
            factory_it->second();
            is_created = true;
            break;
        }
        op_type_info = op_type_info->parent;
    }

    if (!is_created) {
        THROW_IE_EXCEPTION << "Operation: " << op->get_friendly_name()
                           << " of type " << op->get_type_name()
                           << "(op::v" << op->get_type_info().version << ") is not supported";
    }
}

std::vector<cldnn::primitive_id> Program::GetInputPrimitiveIDs(const std::shared_ptr<ngraph::Node>& op) const {
    if (!op) {
        return {};
    }

    std::vector<cldnn::primitive_id> inputPrimitives;
    for (size_t i = 0; i < op->get_input_size(); i++) {
        auto prevOp = op->get_input_node_ptr(i);
        std::string prevName = layer_type_name_ID(prevOp);
        if (prevOp->get_output_size() > 1) {
            prevName += "." + std::to_string(op->get_input_source_output(i).get_index());
        }

        if (!queryMode) {
            if (primitiveIDs.find(prevName) == primitiveIDs.end()) {
                THROW_IE_EXCEPTION << "Input " << prevName << " hasn't been found in primitiveIDs map";
            }
            inputPrimitives.push_back(primitiveIDs.at(prevName));
        } else {
            inputPrimitives.push_back(prevName);
        }
    }
    return inputPrimitives;
}

void Program::AddPrimitiveToProfiler(const std::shared_ptr<ngraph::Node>& op,
                                     cldnn::primitive_id customOutputId) {
    auto id = layer_type_name_ID(op);
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddPrimitiveToProfiler(cldnn::primitive_id id, const std::shared_ptr<ngraph::Node>& op,
                                     cldnn::primitive_id customOutputId) {
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = customOutputId.empty() ? id : customOutputId;
    profilingIDs.push_back(id);
}

void Program::AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                          const std::shared_ptr<ngraph::Node>& op) {
    InitProfileInfo(id, layer_type_lower(op), false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, parentId);
    primitivesToIRLayersMap[id] = { op->get_friendly_name() };
    primitiveIDs[id] = id;
    profilingIDs.push_back(id);
}

void Program::InitProfileInfo(const std::string& layerName,
                              const std::string& layerType,
                              bool isCPU,
                              InferenceEngine::InferenceEngineProfileInfo::LayerStatus status, std::string parentId) {
    std::string layer_type_lower = layerType;
    for (auto& c : layer_type_lower)
        c = tolower(c);

    std::string name = layerName;
    if (name.find(layer_type_lower + ":") != std::string::npos) {
        name = layerName.substr(layerName.find(":") + 1, layerName.length());
    }

    perfMap[layer_type_lower + ":" + name].first = name;
    auto& perfEntry = perfMap[layer_type_lower + ":" + name].second;
    perfEntry.layerType = layerType;
    perfEntry.status = status;
    perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
    perfEntry.isCPU = isCPU;
    perfEntry.parentPrimitive = parentId;
}

}  // namespace CLDNNPlugin
