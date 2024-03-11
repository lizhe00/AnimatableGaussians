#include <torch/torch.h>
#include <vector>

#include "point_mesh.h"

void near_far_smpl(at::Tensor vertices, at::Tensor ray_o, at::Tensor ray_d, at::Tensor near, at::Tensor far, at::Tensor intersect_flag, const float radius);
void nearest_face(at::Tensor vertices, at::Tensor faces, at::Tensor queries, at::Tensor dist, at::Tensor face_ids, at::Tensor nearest_pts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("near_far_smpl", &near_far_smpl, "Near and far computed by SMPL (CUDA)");
    m.def("nearest_face", &nearest_face, "Search nearest face given a mesh (CUDA)");
    m.def("nearest_face_pytorch3d", &PointFaceDistanceForward, "Search nearest face given a mesh (CUDA)");
}

