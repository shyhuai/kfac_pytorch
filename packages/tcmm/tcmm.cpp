#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "tcmm_kernel.h"


std::vector<torch::Tensor> f_symeig(torch::Tensor a) {
    auto c = tcmm_symeig(a);
    return c;
}

std::vector<torch::Tensor> f_symeig_sparse(torch::Tensor a) {
    auto c = tcmm_symeig_sparse(a);
    return c;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("f_symeig", &f_symeig, "TCMM: Eigendecomposition using cuSolver");
    m.def("f_symeig_sparse", &f_symeig_sparse, "TCMM: Eigendecomposition using cuSolverSP");
}
