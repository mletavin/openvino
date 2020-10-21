// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "api/roi_pooling.hpp"

namespace CLDNNPlugin {

static cldnn::pooling_mode GetPoolingMode(std::string method) {
    if (method == "bilinear")
        return cldnn::pooling_mode::bilinear;
    else if (method == "max")
        return cldnn::pooling_mode::max;
    else if (method == "average")
        return cldnn::pooling_mode::average;
    else
        return cldnn::pooling_mode::deformable_bilinear;
}

void Program::CreateDeformablePSROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::DeformablePSROIPooling>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2, 3});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::pooling_mode mode = GetPoolingMode(op->get_mode());
    float trans_std = op->get_trans_std();
    int part_size = op->get_part_size();
    bool no_trans = op->get_input_size() == 2 ? true : false;

    // temporary workaround due to incorrect usage of group_size in the nGraph operation for the DeformablePSROIPooling
    int pooled_width = op->get_group_size();
    int pooled_height = op->get_group_size();
    int group_size = op->get_group_size();
    int output_dim = op->get_output_dim();
    float spatial_scale = op->get_spatial_scale();
    int spatial_bins_x = op->get_spatial_bins_x();
    int spatial_bins_y = op->get_spatial_bins_y();
    bool position_sensitive = true;

    auto psROIPoolingPrim = cldnn::roi_pooling(layerName,
                                                inputPrimitives,
                                                mode,
                                                position_sensitive,
                                                pooled_width,
                                                pooled_height,
                                                spatial_scale,
                                                trans_std,
                                                no_trans,
                                                part_size,
                                                group_size,
                                                output_dim,
                                                spatial_bins_x,
                                                spatial_bins_y);
    topology.add(psROIPoolingPrim);
}

void Program::CreatePSROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::PSROIPooling>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::pooling_mode mode = GetPoolingMode(op->get_mode());
    int group_size = op->get_group_size();
    int output_dim = op->get_output_dim();
    float spatial_scale = op->get_spatial_scale();
    int spatial_bins_x = op->get_spatial_bins_x();
    int spatial_bins_y = op->get_spatial_bins_y();
    bool position_sensitive = true;

    auto psROIPoolingPrim = cldnn::roi_pooling(layerName,
                                               inputPrimitives[0],  // input data
                                               inputPrimitives[1],  // input rois
                                               mode,
                                               position_sensitive,
                                               group_size,
                                               group_size,
                                               spatial_scale,
                                               output_dim,
                                               spatial_bins_x,
                                               spatial_bins_y);
    topology.add(psROIPoolingPrim);
    AddPrimitiveToProfiler(op);
}

void Program::CreateROIPoolingOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::ROIPooling>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {2});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    // params
    auto out_size = op->get_output_size();
    int pooled_height = out_size[0];
    int pooled_width = out_size[1];
    float spatial_scale = op->get_spatial_scale();
    bool position_sensitive = false;

    cldnn::pooling_mode mode = GetPoolingMode(op->get_method());
    auto roiPoolingPrim = cldnn::roi_pooling(layerName,
                                             inputPrimitives[0],  // input data
                                             inputPrimitives[1],  // input rois
                                             mode,
                                             position_sensitive,
                                             pooled_width,
                                             pooled_height,
                                             spatial_scale);

    topology.add(roiPoolingPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
