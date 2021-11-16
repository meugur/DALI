// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/pipeline/operator/builtin/make_contiguous.h"

namespace dali {

void MakeContiguousCPU::RunImpl(HostWorkspace &ws) {
  const auto start = std::chrono::high_resolution_clock::now();

  auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  int batch_size = input.ntensor();
  output.SetLayout(input.GetLayout());
  auto shapes = input.shape();

  auto &thread_pool = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    thread_pool.AddWork([sample_id, &input, &output] (int tid) {
      // HostWorkspace doesn't have any stream
      cudaStream_t stream = 0;
      output[sample_id].Copy(input[sample_id], stream);
    }, shapes.tensor_size(sample_id));
  }
  thread_pool.RunAll();

  const auto end = std::chrono::high_resolution_clock::now();

  // Profile
  for (int i = 0; i < batch_size; i++) {
    const auto &in = ws.Input<CPUBackend>(0, i);
    auto &out = ws.Output<GPUBackend>(0, i);

    auto op_times = in.GetOpTimes();
    op_times["MakeContiguousCPU"] = std::make_pair(start, end);

    out.SetOpTimes(op_times);
    out.SetSourceInfo(in.GetSourceInfo());
  }
}

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousCPU, CPU);

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation, more suitable for execution on the GPU)code")
  .NumInput(1)
  .NumOutput(1)
  .MakeInternal();

}  // namespace dali
