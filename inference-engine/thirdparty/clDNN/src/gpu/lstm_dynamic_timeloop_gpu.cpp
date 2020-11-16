/*
// Copyright (c) 2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "lstm_dynamic_timeloop_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "lstm_dynamic/lstm_dynamic_timeloop_kernel_selector.h"
#include "lstm_dynamic/lstm_dynamic_timeloop_kernel_base.h"
#include "network_impl.h"
#include "error_handler.h"

namespace cldnn {
namespace gpu {

struct lstm_dynamic_timeloop_gpu : typed_primitive_gpu_impl<lstm_dynamic_timeloop> {
    using parent = typed_primitive_gpu_impl<lstm_dynamic_timeloop>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<lstm_dynamic_timeloop>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;
        args.inputs = {(memory_impl::cptr) &instance.input_memory(), (memory_impl::cptr) &instance.dyn_length_memory()};
        if (instance.last_hidden_output_term())
            args.inputs.push_back((memory_impl::cptr) &instance.last_hidden_output_memory());
        if (instance.last_cell_output_term())
            args.inputs.push_back((memory_impl::cptr) &instance.last_cell_output_memory());
        args.output = (memory_impl::cptr) &instance.output_memory();
        args.recurrent = (memory_impl::cptr) &instance.recurrent_memory();
        args.hidden = (memory_impl::cptr) (instance.initial_hidden_term() ? &instance.initial_hidden_memory() : nullptr);
        args.cell = (memory_impl::cptr) (instance.initial_cell_term() ? &instance.initial_cell_memory() : nullptr);
        return args;
    }

public:
    static primitive_impl* create(const lstm_dynamic_timeloop_node& arg) {
        auto dlstm_timeloop_params = get_default_params<kernel_selector::lstm_dynamic_timeloop_params>(arg);

        // dyn length
        const auto& dyn_length_tensor = arg.dyn_length().get_output_layout();
        dlstm_timeloop_params.inputs.push_back(convert_data_tensor(dyn_length_tensor));

        // recurrent
        const auto& recurrent_layout = arg.recurrent().get_output_layout();
        dlstm_timeloop_params.recurrent = convert_data_tensor(recurrent_layout);

        dlstm_timeloop_params.direction = arg.direction();

        if (arg.initial_cell_term()) {
            const auto& cell_layout = arg.initial_cell().get_output_layout();
            dlstm_timeloop_params.set_cell(convert_data_tensor(cell_layout));
        }

        if (arg.last_hidden_output_term()) {
            const auto& last_hidden_output_layout = arg.last_hidden_state().get_output_layout();
            dlstm_timeloop_params.set_last_hidden_output(convert_data_tensor(last_hidden_output_layout));
        }

        if (arg.initial_hidden_term()) {
            const auto& hidden_layout = arg.initial_hidden().get_output_layout();
            dlstm_timeloop_params.set_hidden(convert_data_tensor(hidden_layout));
        }

        if (arg.last_cell_output_term()) {
            const auto& last_cell_state_layout = arg.last_cell_state().get_output_layout();
            dlstm_timeloop_params.set_last_cell_output(convert_data_tensor(last_cell_state_layout));
        }

        const auto& prim = arg.get_primitive();
        if (!prim->activations.empty()) {
            auto a_sz = prim->activations.size();
            auto param_sz = prim->activation_params.size();
            if (param_sz) {
                CLDNN_ERROR_NOT_EQUAL(arg.id(),
                    "number of activations",
                    a_sz,
                    "number of activation parameters",
                    param_sz,
                    "activations/parameters num mismatch");
            }
            for (size_t i = 0; i < a_sz; i++) {
                dlstm_timeloop_params.activations.emplace_back(get_kernel_selector_activation_param(prim->activations[i]),
                    param_sz ? prim->activation_params[i].a : 0.0f,
                    param_sz ? prim->activation_params[i].b : 0.0f);
            }
        }

        if (prim->clip > 0.0f) {
            dlstm_timeloop_params.activations.emplace_back(get_kernel_selector_activation_param(activation_func::clamp), -prim->clip, prim->clip);
        }

        dlstm_timeloop_params.SetOffsetOrder(static_cast<int32_t>(arg.offset_order()));
        dlstm_timeloop_params.clip = arg.clip();

        // finially get best kernel
        auto dlstm_timeloop_optional_params =
            get_default_optional_params<kernel_selector::lstm_dynamic_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::lstm_dynamic_timeloop_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(dlstm_timeloop_params, dlstm_timeloop_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lstm_dynamic = new lstm_dynamic_timeloop_gpu(arg, best_kernels[0]);

        return lstm_dynamic;
    }
};

namespace detail {

attach_lstm_dynamic_timeloop_gpu::attach_lstm_dynamic_timeloop_gpu() {
    auto val_fw = lstm_dynamic_timeloop_gpu::create;

    implementation_map<lstm_dynamic_timeloop>::add({
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw},
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
