#include <torch/extension.h>
#include "include/forward.h"
#include "include/backward.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Compound eyes forward (CUDA)");
  m.def("backward", &backward, "Compound eyes backward (CUDA)");
}