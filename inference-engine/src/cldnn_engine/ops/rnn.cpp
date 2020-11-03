// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/op/lstm_sequence.hpp"

#include "api/reshape.hpp"
#include "api/reorder.hpp"
#include "api/fully_connected.hpp"
#include "api/lstm.hpp"
#include "api/crop.hpp"
#include "api/concatenation.hpp"

namespace CLDNNPlugin {
cldnn::activation_func activation_from_name(std::string name) {
    static const std::map<std::string, cldnn::activation_func> name_mapping = {
        {"sigmoid", cldnn::activation_func::logistic},
        {"tanh", cldnn::activation_func::hyperbolic_tan},
        {"relu", cldnn::activation_func::relu},
    };
    auto itr = name_mapping.find(name);
    if (itr != name_mapping.end())
        return itr->second;
    else
        return cldnn::activation_func::none;
}

void Program::CreateLSTMCellOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::LSTMCell>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {6});
    int lstm_batch_size, lstm_input_size, lstm_hidden_size;
    bool hasBias = true;
    auto inputPrimitives = GetInputPrimitiveIDs(op);

    std::string layerName = layer_type_name_ID(op);
    cldnn::primitive_id weightID = inputPrimitives[3];
    cldnn::primitive_id recurrentID = inputPrimitives[4];
    cldnn::primitive_id biasID = inputPrimitives[5];

    /* check incoming CNN layer and setup required variables */
    {
        const auto in_dims0 = op->get_input_shape(0);
        const auto out_dims0 = op->get_output_shape(0);

        if (in_dims0.size() != 2 ||
            op->get_input_shape(1).size() != 2 ||
            op->get_input_shape(2).size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for LSTMCell op " << op->get_friendly_name();

        lstm_input_size = in_dims0.back();
        lstm_batch_size = in_dims0.at(in_dims0.size()-2);
        lstm_hidden_size = out_dims0.back();
    }

    std::vector<cldnn::activation_func> activations = { cldnn::activation_func::logistic,
                                                        cldnn::activation_func::hyperbolic_tan,
                                                        cldnn::activation_func::hyperbolic_tan };
    std::vector<cldnn::activation_additional_params> activation_params = {};
    auto op_activations = op->get_activations();
    if (!op_activations.empty()) {
        if (op_activations.size() != 3)
            THROW_IE_EXCEPTION << "Wrong number of activations for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            auto af = activation_from_name(op_activations[i]);
            if (af == cldnn::activation_func::none)
                THROW_IE_EXCEPTION << "Wrong or unsupported activation type " << op_activations[i]
                                   << " for LSTMCell op " << op->get_friendly_name();
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 3 || op_b.size() != 3)
            THROW_IE_EXCEPTION << "Wrong number of activation parameters for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
    float clip = op->get_clip();

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstm_dtype = DataTypeFromPrecision(op->get_output_element_type(0));

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id gemmReshapeID = layerName + "_gemmReshape";
    cldnn::primitive_id gemmReorderID = layerName + "_gemmReorder";
    cldnn::primitive_id input_concatID = layerName + "_inputConcat";

    cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
    cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
    cldnn::layout hiddenLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inStateShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    AddInnerPrimitiveToProfiler(inReshapeID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(permuteID, op->get_friendly_name(), op);

    std::string hiddenInResh = inHiddenReshapeID + "_1";
    std::string hiddenInStr = inHiddenReorderID + "_1";
    std::string cellInResh = inHiddenReshapeID + "_2";
    std::string cellInStr = inHiddenReorderID + "_2";
    topology.add(cldnn::reshape(hiddenInResh, inputPrimitives[1], inStateShape));
    topology.add(cldnn::reorder(hiddenInStr, hiddenInResh, hiddenLayout));
    topology.add(cldnn::reshape(cellInResh, inputPrimitives[2], inStateShape));
    topology.add(cldnn::reorder(cellInStr, cellInResh, hiddenLayout));
    topology.add(cldnn::concatenation(input_concatID, { permuteID, hiddenInStr }, cldnn::concatenation::concatenation_axis::along_x));

    AddInnerPrimitiveToProfiler(hiddenInResh, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(hiddenInStr, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(cellInResh, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(cellInStr, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(input_concatID, op->get_friendly_name(), op);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_fc_id = layerName + "_fully_connected";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    cldnn::primitive_id WRconcatID = weightID + "_" + recurrentID;
    topology.add(cldnn::concatenation(WRconcatID, { weightID, recurrentID }, cldnn::concatenation::concatenation_axis::along_f));
    AddInnerPrimitiveToProfiler(WRconcatID, op->get_friendly_name(), op);

    topology.add(cldnn::fully_connected(lstm_fc_id, input_concatID, WRconcatID, hasBias ? biasID : ""));
    topology.add(cldnn::reshape(gemmReshapeID, lstm_fc_id, gemmSz));
    topology.add(cldnn::reorder(gemmReorderID, gemmReshapeID, gemmLayout));
    topology.add(cldnn::lstm_elt(lstm_elt_id, gemmReorderID, cellInStr,
                                 clip, 0, activations, activation_params, cldnn::lstm_weights_order::fizo));

    AddInnerPrimitiveToProfiler(lstm_fc_id, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(gemmReshapeID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(gemmReorderID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(lstm_elt_id, op->get_friendly_name(), op);

    cldnn::primitive_id outputHiddenID = layerName + ".0";
    topology.add(cldnn::crop(outputHiddenID, lstm_elt_id, hiddenSz, cldnn::tensor{0, 0, 0, 0}));
    AddInnerPrimitiveToProfiler(outputHiddenID, op->get_friendly_name(), op);
    cldnn::primitive_id outputCellID = layerName + ".1";
    topology.add(cldnn::crop(outputCellID, lstm_elt_id, hiddenSz, cellCropSz));
    AddInnerPrimitiveToProfiler(outputCellID, op->get_friendly_name(), op);

    // output primitive IDs
    primitiveIDs[outputHiddenID] = outputHiddenID;     // LSTMCell:LSTMCell - "concat hidden"
    primitiveIDs[layerName] = outputHiddenID;          // LSTMCell:LSTMCell:0 - hidden state
    primitiveIDs[outputCellID] = outputCellID;         // LSTMCell:LSTMCell:1 - cell state

    AddPrimitiveToProfiler(layerName, op, outputHiddenID);
}

void Program::CreateLSTMSequenceOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v5::LSTMSequence>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {7});
    
    std::string layerName = layer_type_name_ID(op);
    int lstm_batch_size, lstm_input_size, lstm_hidden_size, lstm_sequence_len;

    auto inputPrimitives = GetInputPrimitiveIDs(op);
    cldnn::primitive_id weightID = inputPrimitives[4];
    cldnn::primitive_id recurrentID = inputPrimitives[5];
    cldnn::primitive_id biasID = inputPrimitives[6];

    {
        const auto in_dims0 = op->get_input_shape(0);
        const auto out_dims0 = op->get_output_shape(0);

        if (in_dims0.size() != 3 ||
            op->get_input_shape(1).size() != 3 ||
            op->get_input_shape(2).size() != 3)
            THROW_IE_EXCEPTION << "Wrong input shapes for LSTMSequence op " << op->get_friendly_name();

        lstm_input_size = in_dims0.back();
        lstm_sequence_len = in_dims0.at(in_dims0.size()-2);
        lstm_batch_size = in_dims0.at(in_dims0.size()-3);
        lstm_hidden_size = out_dims0.back();
    }

    std::vector<cldnn::activation_func> activations = { cldnn::activation_func::logistic,
                                                        cldnn::activation_func::hyperbolic_tan,
                                                        cldnn::activation_func::hyperbolic_tan };
    std::vector<cldnn::activation_additional_params> activation_params = {};
    auto op_activations = op->get_activations();
    if (!op_activations.empty()) {
        if (op_activations.size() != 3)
            THROW_IE_EXCEPTION << "Wrong number of activations for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            auto af = activation_from_name(op_activations[i]);
            if (af == cldnn::activation_func::none)
                THROW_IE_EXCEPTION << "Wrong or unsupported activation type " << op_activations[i]
                                   << " for LSTMCell op " << op->get_friendly_name();
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 3 || op_b.size() != 3)
            THROW_IE_EXCEPTION << "Wrong number of activation parameters for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
    float clip = op->get_clip();

    // auto dir = op->get_direction();
    bool isForward = op->get_direction() == ngraph::op::RecurrentSequenceDirection::FORWARD;

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstm_dtype = DataTypeFromPrecision(op->get_output_element_type(0));

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id inHiddenStateID = inHiddenReshapeID + "_1";
    cldnn::primitive_id inCellStateID = inHiddenReshapeID + "_2";

    std::vector<cldnn::primitive_id> output_ids_offsets;

    cldnn::tensor inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
    cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    topology.add(cldnn::reshape(inHiddenStateID, inputPrimitives[1], inStateShape));
    topology.add(cldnn::reshape(inCellStateID, inputPrimitives[2], inStateShape));

    AddInnerPrimitiveToProfiler(inReshapeID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(permuteID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(inHiddenStateID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(inCellStateID, op->get_friendly_name(), op);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
    cldnn::primitive_id hiddenStr = inHiddenReshapeID+"_1";
    cldnn::primitive_id cellStr = inHiddenReshapeID+"_2";
    cldnn::primitive_id inputCropID = layerName + "_inputCrop";

    cldnn::primitive_id WRconcatID = weightID + "_" + recurrentID;
    topology.add(cldnn::concatenation(WRconcatID, { weightID, recurrentID }, cldnn::concatenation::concatenation_axis::along_y));
    AddInnerPrimitiveToProfiler(WRconcatID, op->get_friendly_name(), op);

    std::vector<size_t> WRreshapeSize = { 4 * size_t(lstm_hidden_size), size_t(lstm_input_size + lstm_hidden_size) };
    cldnn::primitive_id WRreshapeID = WRconcatID + "_reshape";
    auto reshapeInPrim = cldnn::reshape(WRreshapeID, WRconcatID, CldnnTensorFromIEDims(WRreshapeSize));
    topology.add(reshapeInPrim);
    AddInnerPrimitiveToProfiler(WRreshapeID, op->get_friendly_name(), op);

    for (int i = 0; i < lstm_sequence_len; ++i) {
        const std::string id_str = std::to_string(i);
        cldnn::primitive_id concatID = layerName + "_inputConcat" + id_str;
        cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected" + id_str;
        cldnn::primitive_id lstm_fc_resh_id = layerName + "_gemmReshape" + id_str;
        cldnn::primitive_id lstm_fc_reor_id = layerName + "_gemmReorder" + id_str;
        cldnn::primitive_id lstm_elt_id = layerName + "_lstm_elt" + id_str;
        cldnn::primitive_id crop_id = layerName + "_crop" + id_str;

        int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
        const std::string seqIdx_str = std::to_string(seqIdx);

        cldnn::tensor crop_tensor{ inputShape.batch[0], 1, inputShape.spatial[0], inputShape.spatial[1] };
        cldnn::tensor offset_tensor{ 0, static_cast<cldnn::tensor::value_type>(seqIdx), 0, 0 };
        cldnn::primitive_id inputCrop_id = inputCropID + ":" + seqIdx_str;
        topology.add(cldnn::crop(inputCrop_id, permuteID, crop_tensor, offset_tensor));
        AddInnerPrimitiveToProfiler(inputCrop_id, op->get_friendly_name(), op);

        topology.add(cldnn::concatenation(concatID, { inputCrop_id, hiddenStr }, cldnn::concatenation::concatenation_axis::along_x));
        AddInnerPrimitiveToProfiler(concatID, op->get_friendly_name(), op);
        topology.add(cldnn::fully_connected(lstm_fc_id, concatID, WRreshapeID, biasID));
        AddInnerPrimitiveToProfiler(lstm_fc_id, op->get_friendly_name(), op);

        topology.add(cldnn::reshape(lstm_fc_resh_id, lstm_fc_id, gemmSz));
        topology.add(cldnn::reorder(lstm_fc_reor_id, lstm_fc_resh_id, gemmLayout));
        topology.add(cldnn::lstm_elt(lstm_elt_id, lstm_fc_reor_id, cellStr,
                                     clip, 0, activations, activation_params, cldnn::lstm_weights_order::fizo));
        AddInnerPrimitiveToProfiler(lstm_fc_resh_id, op->get_friendly_name(), op);
        AddInnerPrimitiveToProfiler(lstm_fc_reor_id, op->get_friendly_name(), op);
        AddInnerPrimitiveToProfiler(lstm_elt_id, op->get_friendly_name(), op);

        hiddenStr = crop_id + ":hidden";
        cellStr = crop_id + ":cell";
        topology.add(cldnn::crop(hiddenStr, lstm_elt_id, hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }));
        AddInnerPrimitiveToProfiler(hiddenStr, op->get_friendly_name(), op);
        output_ids_offsets.push_back(hiddenStr);

        if (i < lstm_sequence_len - 1) {
            topology.add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
            AddInnerPrimitiveToProfiler(cellStr, op->get_friendly_name(), op);
        } else {
            // last hidden state crop (output 2)
            cldnn::primitive_id outputHiddenID = layerName + ".1";
            primitiveIDs[hiddenStr] = hiddenStr;
            primitiveIDs[outputHiddenID] = hiddenStr;

            // last cell state crop (output 3)
            topology.add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
            cldnn::primitive_id outputCellID = layerName + ".2";
            AddInnerPrimitiveToProfiler(cellStr, op->get_friendly_name(), op);
            primitiveIDs[outputCellID] = cellStr;
        }
    }

    if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());
    // concatenated hidden state (output 1)
    cldnn::primitive_id outputConcatID = layerName + ".0";
    cldnn::primitive_id concatStr = layerName + ":hiddenConcat";
    topology.add(cldnn::concatenation(concatStr, output_ids_offsets, cldnn::concatenation::along_f));

    primitiveIDs[outputConcatID] = concatStr;
    primitiveIDs[layerName] = concatStr;
    AddPrimitiveToProfiler(layerName, op);
}

}  // namespace CLDNNPlugin
