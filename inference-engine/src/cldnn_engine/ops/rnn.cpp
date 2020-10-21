// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/reshape.hpp"
#include "api/reorder.hpp"
#include "api/fully_connected.hpp"
#include "api/lstm.hpp"
#include "api/crop.hpp"
#include "api/concatenation.hpp"

namespace CLDNNPlugin {

void Program::CreateLSTMCellOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v4::LSTMCell>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {6});
    int lstm_batch_size, lstm_input_size, lstm_hidden_size;
    bool hasBias = false;
    auto inputPrimitives = GetInputPrimitiveIDs(op);

    std::string layerName = layer_type_name_ID(op);
    cldnn::primitive_id weightID = inputPrimitives[3];
    cldnn::primitive_id biasID = inputPrimitives[5];

    /* check incoming CNN layer and setup required variables */
    {
        const auto in_dims0 = op->get_input_shape(0);
        const auto out_dims0 = op->get_output_shape(0);

        lstm_input_size = in_dims0.back();
        lstm_batch_size = in_dims0.at(in_dims0.size()-2);
        lstm_hidden_size = out_dims0.back();

        if (in_dims0.size() != 2 ||
            op->get_input_shape(1).size() != 2 ||
            op->get_input_shape(2).size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for LSTMCell op " << op->get_friendly_name();
    }

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id gemmReshapeID = layerName + "_gemmReshape";
    cldnn::primitive_id gemmReorderID = layerName + "_gemmReorder";
    cldnn::primitive_id concatID = layerName + "_inputConcat";

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstmPrecision = op->get_output_element_type(0);

    cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, inputShape);
    cldnn::layout hiddenLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, hiddenStateShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    AddInnerPrimitiveToProfiler(inReshapeID, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(permuteID, op->get_friendly_name(), op);

    std::string hiddenInResh = inHiddenReshapeID + "_1";
    std::string hiddenInStr = inHiddenReorderID + "_1";
    std::string cellInResh = inHiddenReshapeID + "_2";
    std::string cellInStr = inHiddenReorderID + "_2";
    topology.add(cldnn::reshape(hiddenInResh, inputPrimitives[1], hiddenStateShape));
    topology.add(cldnn::reorder(hiddenInStr, hiddenInResh, hiddenLayout));
    topology.add(cldnn::reshape(cellInResh, inputPrimitives[2], hiddenStateShape));
    topology.add(cldnn::reorder(cellInStr, cellInResh, hiddenLayout));
    topology.add(cldnn::concatenation(concatID, { permuteID, hiddenInStr }, cldnn::concatenation::concatenation_axis::along_x));

    AddInnerPrimitiveToProfiler(hiddenInResh, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(hiddenInStr, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(cellInResh, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(cellInStr, op->get_friendly_name(), op);
    AddInnerPrimitiveToProfiler(concatID, op->get_friendly_name(), op);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_fc_id = layerName + "_fully_connected";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    topology.add(cldnn::fully_connected(lstm_fc_id, concatID, weightID, hasBias ? biasID : ""));
    topology.add(cldnn::reshape(gemmReshapeID, lstm_fc_id, gemmSz));
    topology.add(cldnn::reorder(gemmReorderID, gemmReshapeID, gemmLayout));
    topology.add(cldnn::lstm_elt(lstm_elt_id, gemmReorderID, cellInStr,
                                    0, 0, {}, {}, cldnn::lstm_weights_order::fizo));

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

    THROW_IE_EXCEPTION << "Unimplemented yet";
}

}  // namespace CLDNNPlugin
