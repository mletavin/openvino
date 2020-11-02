// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <cstdint>

#include <ie_icnn_network.hpp>
#include "details/ie_exception.hpp"

#include "cldnn_config.h"

#include <api/engine.hpp>
#include <api/topology.hpp>

#include "ngraph/ngraph.hpp"

#define INVALID_OP_MESSAGE std::string("Invalid ngraph Node type passed into ") + __PRETTY_FUNCTION__

// Forward declarations for cldnn primitive parameters
namespace cldnn {
enum class activation_func;
struct activation_additional_params;
enum class reduce_mode : uint16_t;
enum class eltwise_mode : int32_t;
}  // namespace cldnn

namespace CLDNNPlugin {

inline std::string layer_type_lower(const ngraph::Node* op) {
    std::string layerType = op->get_type_name();
    std::transform(layerType.begin(), layerType.end(), layerType.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return layerType;
}

inline std::string layer_type_name_ID(const ngraph::Node* op) {
    return layer_type_lower(op) + ":" + op->get_friendly_name();
}

inline std::string layer_type_lower(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_lower(op.get());
}

inline std::string layer_type_name_ID(const std::shared_ptr<ngraph::Node>& op) {
    return layer_type_name_ID(op.get());
}

struct PerfCounter {
    InferenceEngine::InferenceEngineProfileInfo::LayerStatus status;
    bool isCPU;
    uint64_t realTime_uSec;
    uint64_t cpu_uSec;
    uint32_t num;
    std::string layerType;
    std::string parentPrimitive;

public:
    PerfCounter() : realTime_uSec(0), cpu_uSec(0), num(0),
                    status(InferenceEngine::InferenceEngineProfileInfo::NOT_RUN), isCPU(false) {}

    long long realTime_avg() const { return (num == 0) ? 0 : realTime_uSec / num; }
    long long cpu_avg() const { return (num == 0) ? 0 : cpu_uSec / num; }
};

class Program {
public:
    Program(InferenceEngine::ICNNNetwork& network, std::shared_ptr<const cldnn::engine> engine, const Config& config);
    Program() : m_config({}), m_engine(nullptr), m_curBatch(-1), queryMode(false) {}
    std::shared_ptr<cldnn::program> getCompiledProgram(int program_id = 0);

    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<cldnn::primitive_id, std::vector<std::string>> primitivesToIRLayersMap;
    std::map<cldnn::primitive_id, std::string> IRToNgraphLayersMap;
    std::map<std::string, std::vector<cldnn::primitive_id>> prevPrimitiveIDs;
    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;

    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<std::string, InferenceEngine::SizeVector> outputDims;
    std::map<std::string, cldnn::layout> inputLayouts;
    std::map<const char *, cldnn::primitive_id> blobMemCache;

    int m_max_batch;
    int m_curBatch;

    const std::map<std::string, cldnn::layout>& getInputLayouts() const { return inputLayouts; }
    int GetMaxBatchSizeForSingleProgram();

    bool IsOpSupported(const InferenceEngine::ICNNNetwork& network, const std::shared_ptr<ngraph::Node>& op);

private:
    std::vector<std::shared_ptr<cldnn::program>> m_programs;
    std::shared_ptr<const cldnn::engine> m_engine;
    Config m_config;

    bool queryMode;

    static const cldnn::primitive_id m_preProcessTag;
    static const cldnn::primitive_id m_meanValuesTag;
    static const cldnn::primitive_id m_workaroundTag;
    static const cldnn::primitive_id m_preCustomLayerTag;
    static const cldnn::primitive_id m_postCustomLayerTag;

    void EnableQueryMode() { queryMode = true; }
    void DisableQueryMode() { queryMode = false; }

    std::shared_ptr<cldnn::program> BuildProgram(std::vector<std::shared_ptr<ngraph::Node>> ops,
                                                 InferenceEngine::InputsDataMap networkInputs,
                                                 InferenceEngine::OutputsDataMap networkOutputs);

    // Profiling utils
    void InitProfileInfo(const std::string& layerName,
                         const std::string& layerType,
                         bool isCPU = false,
                         InferenceEngine::InferenceEngineProfileInfo::LayerStatus status
                         = InferenceEngine::InferenceEngineProfileInfo::EXECUTED,
                         std::string parentId = "");
    void AddPrimitiveToProfiler(cldnn::primitive_id id, const std::shared_ptr<ngraph::Node>& op,
                                cldnn::primitive_id customOutputId = "");
    void AddPrimitiveToProfiler(const std::shared_ptr<ngraph::Node>& op,
                                cldnn::primitive_id customOutputId = "");
    void AddInnerPrimitiveToProfiler(cldnn::primitive_id id, cldnn::primitive_id parentId,
                                     const std::shared_ptr<ngraph::Node>& op);

    // Graph construction helpers
    void ValidateInputs(const std::shared_ptr<ngraph::Node>& op, std::vector<size_t> validInputsCount);
    std::vector<cldnn::primitive_id> GetInputPrimitiveIDs(const std::shared_ptr<ngraph::Node>& op) const;

    void Load(InferenceEngine::ICNNNetwork &network);
    void CreateSingleLayerPrimitive(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& op,
                                    InferenceEngine::InputsDataMap networkInputs,
                                    InferenceEngine::OutputsDataMap networkOutputs);
    bool CanProcessDynBatch(std::vector<std::shared_ptr<ngraph::Node>> ops, InferenceEngine::InputsDataMap networkInputs) const;
    void changeInputBatch(int batch);

    // Creators for ngraph ops
    void CreateParameterOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node, InferenceEngine::InputsDataMap networkInputs);
    void CreateResultOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node, InferenceEngine::OutputsDataMap networkOutputs);

    void CreateConstantOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    void CreateConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGroupConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateConvolutionBackpropDataOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGroupConvolutionBackpropDataOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateDeformableConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateBinaryConvolutionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Pooling
    void CreateMaxPoolOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAvgPoolOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Unary
    void CreateUnaryEltwiseOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node,
                              cldnn::activation_func func, cldnn::activation_additional_params params);

    void CreateTanhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateEluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePReluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateClampOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateExpOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAsinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAsinhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAcosOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAcoshOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAtanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAtanhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateAbsOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateFloorOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateCeilingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSqrtOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateErfOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateHardSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLogOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateNegativeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSeluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSoftPlusOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateTanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSinhOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateCosOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateCoshOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSwishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateHSwishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMishOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGeluOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSignOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateHSigmoidOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateRoundOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Reduction
    void CreateReduceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node, cldnn::reduce_mode mode, bool keep_dims);
    void CreateReduceLogicalAndOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceLogicalOrOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceMeanOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceMinOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceMaxOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceProdOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceL1Op(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReduceL2Op(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Element-wise
    void CreateElementwiseOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node, cldnn::eltwise_mode mode);
    void CreateAddOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMultiplyOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMaximumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMinimumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSubtractOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateDivideOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSquaredDifferenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateNotEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLessOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLessEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGreaterOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGreaterEqualOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLogicalNotOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLogicalAndOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLogicalOrOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLogicalXorOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePowerOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateFloorModOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Reshape
    void CreateCommonReshapeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReshapeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSqueezeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateUnsqueezeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    void CreateBatchToSpaceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSpaceToBatchOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSpaceToDepthOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateDepthToSpaceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateCumSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateExtractImagePatchesOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateEmbeddingBagOffsetsSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateEmbeddingBagPackedSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateEmbeddingSegmentsSumOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSoftmaxOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateProposalOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateFakeQuantizeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGatherOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGatherTreeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateTransposeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePriorBoxOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePriorBoxClusteredOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePriorBoxClusteredIEOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMatMulOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateTopKOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateShuffleChannelsOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateDetectionOutputOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateConcatOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePSROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateDeformablePSROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateStridedSliceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateTileOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreatePadOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateOneHotOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateNonMaxSuppressionOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSelectOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateCTCGreedyDecoderOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateInterpolateOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReverseSequenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateScatterUpdateOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // RNN
    void CreateLSTMSequenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLSTMCellOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Broadcast
    void CreateCommonBroadcastOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateBroadcastOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Split
    void CreateCommonSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateVariadicSplitOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Convert
    void CreateConvertOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateConvertLikeOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Normalization
    void CreateNormalizeL2Op(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateMVNOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateGRNOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateLRNOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Yolo
    void CreateRegionYoloOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);
    void CreateReorgYoloOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node);

    // Custom
    void CreateCustomOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node, CLDNNCustomLayerPtr customLayer);
};

}  // namespace CLDNNPlugin
