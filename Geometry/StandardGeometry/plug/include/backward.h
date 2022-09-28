#include <torch/extension.h>

torch::Tensor backward(torch::Tensor sino, torch::Tensor _angles, torch::Tensor _volumeSize, torch::Tensor _detectorSize, const float sid, const float sdd, const int device);