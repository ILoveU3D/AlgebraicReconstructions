#include <torch/extension.h>

torch::Tensor forward(torch::Tensor volume, torch::Tensor angles, torch::Tensor _volumeSize, torch::Tensor _detectorSize, const float sid, const float sdd, const long device, float sampleInterval = -1, float sliceInterval = -1);