#include <ATen/ATen.h>
#include <torch/torch.h>


void root_finding(at::Tensor _weight_volume,  // (X, Y, Z, J)
    at::Tensor _grad_volume, // (X, Y, Z, J*3)
    at::Tensor _xt,  // (B, N, 3)
    at::Tensor _xc_init,  // (B, N, 3)
    at::Tensor _jnt_mats,  // (B, J, 4, 4)
    at::Tensor _bounds, // (2, 3)
    at::Tensor _res, // (3)
    at::Tensor _xc_opt,
    float _lambda,
    int _iter_num);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("root_finding", &root_finding);
}

