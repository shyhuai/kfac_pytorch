#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "tcmm_kernel.h"


std::vector<torch::Tensor> f_syseig(torch::Tensor a) {
    auto c = tcmm_syseig(a);
    return c;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_syseig", &f_syseig, "TCMM: Eigendecomposition using cuSolver");
}
