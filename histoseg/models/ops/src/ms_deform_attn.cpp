#include <torch/extension.h>

#ifdef WITH_CUDA
torch::Tensor ms_deform_attn_cuda_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step);

torch::Tensor ms_deform_attn_cuda_backward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    const int im2col_step);
#endif

torch::Tensor ms_deform_attn_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step) {
  if (value.type().is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index,
                                       sampling_loc, attn_weight, im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

torch::Tensor ms_deform_attn_backward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    const int im2col_step) {
  if (value.type().is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index,
                                        sampling_loc, attn_weight, grad_output,
                                        im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward,
        "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward,
        "ms_deform_attn_backward");
}
