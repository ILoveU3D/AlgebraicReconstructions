#include <torch/extension.h>

torch::Tensor forward(torch::Tensor volume, torch::Tensor angles, torch::Tensor _volumeSize, torch::Tensor _detectorSize, float sid, float sdd, float offset, const float pixelSpacing, const long device, float sampleInterval = -1, float sliceInterval = -1);