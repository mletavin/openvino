// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/parameter.hpp"

#include "api/input_layout.hpp"
#include "api/reorder.hpp"
#include "api/data.hpp"

using namespace InferenceEngine;

namespace CLDNNPlugin {

void Program::CreateParameterOp(cldnn::topology& topology,
                                const std::shared_ptr<ngraph::Node>& node,
                                InferenceEngine::InputsDataMap networkInputs) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    if (networkInputs.find(op->get_friendly_name()) == networkInputs.end()) {
        THROW_IE_EXCEPTION << "Can't find input " << op->get_friendly_name() << " in InputsDataMap";
    }
    auto inputInfo = networkInputs.at(op->get_friendly_name());
     // first create and add the input layout
    const auto inputDesc = inputInfo->getTensorDesc();
    const auto inputDims = inputDesc.getDims();
    Layout l = inputDesc.getLayout();
    Precision ip = inputDesc.getPrecision();

    cldnn::format inputFormat = cldnn::format::bfyx;
    if (InferenceEngine::Layout::BLOCKED == l && 6 == inputDims.size()) {
        inputFormat = cldnn::format::bfwzyx;
    } else {
        inputFormat = FormatFromLayout(l);
    }

    cldnn::tensor dataTensor;
    cldnn::tensor::value_type batch = (m_max_batch <= 1)
                                      ? (inputDims.size() > 3 ? TensorValue(inputDims[0]) : 1)
                                      : TensorValue(m_curBatch);
    switch (inputDims.size()) {
    case 6:
        dataTensor = cldnn::tensor(cldnn::batch(batch),
                                   cldnn::feature(inputDims[1]),
                                   cldnn::spatial(inputDims[5], inputDims[4], inputDims[3], inputDims[2]));
        break;
    case 5:
        if (InferenceEngine::Layout::NCDHW == l) {
            dataTensor = cldnn::tensor(cldnn::batch(batch),
                                       cldnn::feature(inputDims[1]),
                                       cldnn::spatial(inputDims[4], inputDims[3], inputDims[2]));
        } else {
            THROW_IE_EXCEPTION  << "Unsupported layout (" << l << ") in 5D input " << inputInfo->name();
        }
        break;
    case 4:
        if (InferenceEngine::Layout::NCHW == l || InferenceEngine::Layout::CHW == l) {
            dataTensor = cldnn::tensor(batch,
                                       TensorValue(inputDims[1]), TensorValue(inputDims[3]), TensorValue(inputDims[2]));
        } else if (InferenceEngine::Layout::NHWC == l) {
            dataTensor = cldnn::tensor(batch,
                                       TensorValue(inputDims[1]), TensorValue(inputDims[3]), TensorValue(inputDims[2]));
        } else {
            THROW_IE_EXCEPTION << "Unsupported layout (" << l << ") in 4D input " + inputInfo->name();
        }
        break;
    case 3:
        if (InferenceEngine::Layout::CHW == l) {
            dataTensor = cldnn::tensor(TensorValue(inputDims[0]), TensorValue(inputDims[1]), 1, TensorValue(inputDims[2]));
        } else {
            THROW_IE_EXCEPTION << "Unsupported layout (" << l << ") in 3D input " + inputInfo->name();
        }
        break;
    case 2:
        if (InferenceEngine::Layout::NCHW == l || InferenceEngine::NC == l) {
            dataTensor = cldnn::tensor(TensorValue(inputDims[0]), TensorValue(inputDims[1]), 1, 1);
        } else {
            THROW_IE_EXCEPTION << "Unsupported layout (" << l << ") in 2D input " << inputInfo->name();
        }
        break;
    case 1:
        dataTensor = cldnn::tensor(TensorValue(inputDims[0]), 1, 1, 1);
        break;
    case 0:
        dataTensor = cldnn::tensor(1, 1, 1, 1);
        break;
    default: THROW_IE_EXCEPTION << "Invalid data dimensions";
    }
    cldnn::layout networkInputLayout(DataTypeFromPrecision(ip),
                                     inputFormat,
                                     dataTensor);

    // look at the expected color format of this input
    auto inputName = layer_type_name_ID(op);
    auto preProcess = inputInfo->getPreProcess();
    size_t meanChannels = preProcess.getNumberOfChannels();
    networkInputLayout.format = inputFormat;
    networkInputLayout.size = networkInputLayout.size.transform(inputFormat, 1);
    networkInputLayout.data_type = DataTypeFromPrecision(op->get_output_element_type(0));
    auto preprocessPrimID = "reorder:" + inputName + m_preProcessTag;
    cldnn::primitive_id meanBlobID = inputName + m_meanValuesTag;
    std::vector<float> meanValues;

    if ((meanChannels > 0) &&
        (meanChannels != networkInputLayout.size.feature[0])) {
        THROW_IE_EXCEPTION << "Mismatched mean values channels in input " << inputName;
    }

    switch (preProcess.getMeanVariant()) {
    case NONE:
    case MEAN_VALUE: {
        if (meanChannels > 0) {
            for (size_t c = 0; c < meanChannels; c++) {
                if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                    THROW_IE_EXCEPTION << "not supporting stdScale yet in input " << inputName;
                meanValues.push_back(preProcess[c]->meanValue);
            }
        }
        break;
    }
    case MEAN_IMAGE: {
        IE_ASSERT(meanChannels);
        // first merge all mean values to a single blob
        // todo make sure mean blob precision is the same as the input precision
        auto meanDims = inputDims;
        // overwrite batches with 1
        switch (meanDims.size()) {
        case 4: meanDims[0] = 1;
            break;
        default:
            THROW_IE_EXCEPTION << "Missing batch dimensions in input image";
        }
        const TensorDesc desc(Precision(Precision::FP32), meanDims, TensorDesc::getLayoutByDims(meanDims));
        InferenceEngine::TBlob<float> meanBlob(desc);
        meanBlob.allocate();
        auto meanBlobData = meanBlob.data();
        for (size_t c = 0; c < meanChannels; c++) {
            if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                THROW_IE_EXCEPTION << "not supporting stdScale yet in input " << inputName;
            auto channelMeanBlob = std::dynamic_pointer_cast<TBlob<float>>(preProcess[c]->meanData);
            auto channelSize = channelMeanBlob->size();
            auto channelBlobData = channelMeanBlob->data();
            for (size_t i = 0; i < channelSize; i++) {
                meanBlobData[(c * channelSize) + i] = channelBlobData[i];
            }
        }
        // then create a data primitive for the mean values
        auto meanBlobPtr = std::make_shared<InferenceEngine::TBlob<float>>(meanBlob);

        // mean values will use external format (sub in the input format before convert to new format)
        cldnn::tensor meanBlobTensor(networkInputLayout.size);
        meanBlobTensor.batch[0] = 1;  // mean values have no batches
        cldnn::layout meanBlobLayout(cldnn::data_types::f32, cldnn::format::bfyx, meanBlobTensor);

        auto data = static_cast<const char *>(meanBlobPtr->buffer());

        auto bufIter = blobMemCache.find(data);
        if (bufIter != blobMemCache.end()) {
            meanBlobID = bufIter->second;
        } else {
            auto mem = cldnn::memory::allocate(*m_engine, meanBlobLayout, 0, false);
            auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
            auto buf = tmpPointer.data();
            auto bufSize = meanBlobLayout.bytes_count();

            std::memcpy(&buf[0], &data[0], bufSize);

            topology.add(cldnn::data(meanBlobID, mem));
            blobMemCache[data] = meanBlobID;
        }
        break;
    }
    default: THROW_IE_EXCEPTION << "Invalid mean variant in input " << inputName;
        break;
    }

    if (ColorFormat::NV12 == preProcess.getColorFormat() && m_config.nv12_two_inputs) {
        // for NV12, create two input layouts with reorder instead of one,
        // and then would expect compound blob in inferRequest
        if (Layout::NCHW != l &&
            (Precision::I8 != ip || Precision::U8 != ip)) {
            THROW_IE_EXCEPTION << "Unsupported layout (" << l << ") or precision "
                               << ip.name() << ") for NV12 input " + inputInfo->name();
        }
        int height = inputDims[2];
        int width = inputDims[3];

        std::string y_name = inputName + "_Y";
        std::string uv_name = inputName + "_UV";

        cldnn::layout y_layout(DataTypeFromPrecision(ip),
                                cldnn::format::nv12, { 1, 1, width, height });
        cldnn::layout uv_layout(DataTypeFromPrecision(ip),
                                cldnn::format::nv12, { 1, 2, width / 2, height / 2 });
        auto inputY = cldnn::input_layout(y_name, y_layout);
        auto inputUV = cldnn::input_layout(uv_name, uv_layout);

        topology.add(inputY);
        inputLayouts.insert({ inputInfo->name() + "_Y", y_layout });
        topology.add(inputUV);
        inputLayouts.insert({ inputInfo->name() + "_UV", uv_layout });
        switch (preProcess.getMeanVariant()) {
        case NONE:
        case MEAN_VALUE: {
            topology.add(cldnn::reorder(preprocessPrimID, y_name, uv_name, networkInputLayout, meanValues));
            break;
        }
        case MEAN_IMAGE: {
            topology.add(cldnn::reorder(preprocessPrimID, y_name, uv_name, networkInputLayout, meanBlobID));
            break;
        }
        default: THROW_IE_EXCEPTION << "Invalid mean variant in input " + inputName;
            break;
        }

        primitivesToIRLayersMap[preprocessPrimID] = { inputInfo->name() };
        primitivesToIRLayersMap[y_name] = { inputInfo->name() };
        primitivesToIRLayersMap[uv_name] = { inputInfo->name() };
        profilingIDs.push_back(preprocessPrimID);
        InitProfileInfo(preprocessPrimID, "Reorder");
    } else {
        cldnn::layout inputLayout(networkInputLayout);
        inputLayout.data_type = DataTypeFromPrecision(ip);
        inputLayouts.insert({ inputInfo->name(), inputLayout });

        topology.add(cldnn::input_layout(inputName, inputLayout));
        primitivesToIRLayersMap[inputName] = { inputInfo->name() };

        switch (preProcess.getMeanVariant()) {
        case NONE:
        case MEAN_VALUE: {
            topology.add(cldnn::reorder(preprocessPrimID, inputName, networkInputLayout, meanValues));
            break;
        }
        case MEAN_IMAGE: {
            topology.add(cldnn::reorder(preprocessPrimID,
                                        inputName,
                                        networkInputLayout,
                                        meanBlobID));
            break;
        }
        default: THROW_IE_EXCEPTION << "Invalid mean variant in input " << inputName;
            break;
        }
        InitProfileInfo(preprocessPrimID, "reorder");
        primitiveIDs[preprocessPrimID] = preprocessPrimID;
        profilingIDs.push_back(preprocessPrimID);
    }

    primitiveIDs[inputName] = preprocessPrimID;
    primitiveIDs[preprocessPrimID] = preprocessPrimID;
}

}  // namespace CLDNNPlugin
