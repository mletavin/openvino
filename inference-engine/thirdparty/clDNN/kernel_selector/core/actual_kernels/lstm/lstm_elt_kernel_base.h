/*
// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>
#include <map>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_elt_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_elt_params : public lstm_gate_params, public base_params {
    lstm_elt_params() : base_params(KernelType::LSTM_ELT) {}

    DataTensor cell;
    bool has_cell = false;
    float clip = 0;
    bool input_forget = false;
    uint32_t direction = 0;
    uint32_t cell_direction = 0;

    void SetCell(const DataTensor& v) {
        cell = v;
        has_cell = true;
    }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        if (has_cell) {
            k.EnableLSTMEltCell();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_elt_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_elt_optional_params : optional_params {
    lstm_elt_optional_params() : optional_params(KernelType::LSTM_ELT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LSTMEltKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LSTMEltKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LSTMEltKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_elt_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;

    bool Validate(const Params& p, const optional_params&) const override {
        if (p.GetType() != KernelType::LSTM_ELT) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
