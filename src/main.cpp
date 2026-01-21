#include <torch/extension.h>

void launch_add(torch::Tensor x, torch::Tensor y, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &launch_add, "CUDA Vector Add");
}