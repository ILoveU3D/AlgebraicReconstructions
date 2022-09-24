#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> cone_project_cuda_forward(
    torch::Tensor volume,
    torch::Tensor ray_vectors,
    int number_of_projections,
    const int volume_width, 
    const int volume_height, 
    const int volume_depth, 
    const float volume_origin_x,
    const float volume_origin_y,
    const float volume_origin_z,
    const int detector_width, 
    const int detector_height, 
    const float detector_origin_x,
    const float detector_origin_y,
    const float sid, 
    const float sdd,
    const float volbiasz,
    const float fbiasz,
    const float dSliceInterval,
    const int   deviceId);


std::vector<torch::Tensor> cone_project_cuda_backward(
                             torch::Tensor sinogram,
                             torch::Tensor ray_vectors,
                             int number_of_projections,
                             const int volume_width, 
                             const int volume_height, 
                             const int volume_depth,
                             const float volume_origin_x, 
                             const float volume_origin_y,
                             const float volume_origin_z,
                             const int detector_width, 
                             const int detector_height, 
                             const float detector_origin_x,
                             const float detector_origin_y,
                             const float sid, 
                             const float sdd,
                             const float volbiasz,
                             const float fbiasz,
                             const float dSliceInterval,
                             const int   deviceId);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ConeProjectZ_forward(
    torch::Tensor volume,
    torch::Tensor ray_vectors,
    int number_of_projections,
    const int volume_width, 
    const int volume_height, 
    const int volume_depth, 
    const float volume_origin_x,
    const float volume_origin_y,
    const float volume_origin_z,
    const int detector_width, 
    const int detector_height, 
    const float detector_origin_x,
    const float detector_origin_y,
    const float sid, 
    const float sdd,
    const float volbiasz,
    const float fbiasz,
    const float dSliceInterval,
    const int   deviceId) {
  CHECK_INPUT(volume);
  CHECK_INPUT(ray_vectors);


  return cone_project_cuda_forward(volume, ray_vectors, number_of_projections, volume_width, volume_height, volume_depth, volume_origin_x,
                         volume_origin_y, volume_origin_z, detector_width, detector_height, detector_origin_x,
                         detector_origin_y, sid, sdd, volbiasz, fbiasz, dSliceInterval, deviceId);
}

std::vector<torch::Tensor> ConeProjectZ_backward(
                             torch::Tensor sinogram,
                             torch::Tensor ray_vectors,
                             int number_of_projections,
                             const int volume_width,
                             const int volume_height,
                             const int volume_depth,
                             const float volume_origin_x,
                             const float volume_origin_y,
                             const float volume_origin_z,
                             const int detector_width,
                             const int detector_height,
                             const float detector_origin_x,
                             const float detector_origin_y,
                             const float sid,
                             const float sdd,
                             const float volbiasz,
                             const float fbiasz,
                             const float dSliceInterval,
                             const int   deviceId) {
  CHECK_INPUT(sinogram);
  CHECK_INPUT(ray_vectors);

  //printf("back ok1\n");
  //printf("after volume_size: %f %f %d %f %f %f\n", volume_origin_x, volume_origin_y, detector_size, detector_origin, sid, sdd);
  //printf("sino[368][0]: %f\n", sinogram[368*360]);

  return cone_project_cuda_backward(sinogram, ray_vectors, number_of_projections, volume_width, volume_height, volume_depth, volume_origin_x,
                         volume_origin_y, volume_origin_z, detector_width, detector_height, detector_origin_x,
                         detector_origin_y, sid, sdd, volbiasz, fbiasz, dSliceInterval, deviceId);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ConeProjectZ_forward, "ConeProjectZ forward (CUDA)");
  m.def("backward", &ConeProjectZ_backward, "ConeProjectZ backward (CUDA)");
}