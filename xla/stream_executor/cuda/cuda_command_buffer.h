/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/scoped_gpu_graph_exec.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

#if CUDA_VERSION < 12030
typedef cuuint64_t CUgraphConditionalHandle;
#endif

namespace stream_executor::gpu {

// This class implements GpuCommandBuffer for Nvidia GPUs.
class CudaCommandBuffer final : public GpuCommandBuffer {
 public:
  // Creates a new CUDA command buffer and the underlying CUDA graph.
  static absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> Create(
      Mode mode, StreamExecutor* parent, CudaContext* cuda_context);

  ~CudaCommandBuffer() override;

 private:
  CudaCommandBuffer(Mode mode, StreamExecutor* parent,
                    CudaContext* cuda_context, CUgraph graph,
                    bool is_owned_graph)
      : GpuCommandBuffer(mode, parent),
        parent_(parent),
        cuda_context_(cuda_context),
        graph_(graph),
        is_owned_graph_(is_owned_graph) {
    VLOG(5) << "Created command buffer for graph " << graph_
            << "; mode=" << absl::StrCat(mode)
            << "; is_owned_graph=" << is_owned_graph_;
  }

  //===--------------------------------------------------------------------===//
  // APIs for launching kernels to update conditional handles.
  //===--------------------------------------------------------------------===//

  absl::StatusOr<GraphNodeHandle> CreateSetCaseConditionNode(
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default,
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status UpdateSetCaseConditionNode(
      GraphNodeHandle handle,
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default) override;

  absl::StatusOr<GraphNodeHandle> CreateSetWhileConditionNode(
      GraphConditionalHandle conditional, DeviceMemory<bool> predicate,
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status UpdateSetWhileConditionNode(
      GraphNodeHandle handle, GraphConditionalHandle conditional,
      DeviceMemory<bool> predicate) override;

  //===--------------------------------------------------------------------===//

  using NoOpKernel = TypedKernel<>;

  absl::StatusOr<NoOpKernel*> GetNoOpKernel();

  absl::StatusOr<GraphConditionalNodeHandle> CreateConditionalNode(
      absl::Span<const GraphNodeHandle> dependencies,
      GraphConditionalHandle conditional, ConditionType type) override;

  absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceMemoryBase destination, BitPattern bit_pattern,
      size_t num_elements) override;

  absl::Status UpdateMemsetNode(GraphNodeHandle node_handle,
                                DeviceMemoryBase destination,
                                BitPattern bit_pattern,
                                size_t num_elements) override;

  absl::StatusOr<GraphNodeHandle> CreateMemcpyD2DNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceMemoryBase destination, DeviceMemoryBase source,
      uint64_t size) override;

  absl::Status UpdateMemcpyD2DNode(GraphNodeHandle node_handle,
                                   DeviceMemoryBase destination,
                                   DeviceMemoryBase source,
                                   uint64_t size) override;

  absl::Status PopulateDnnGraphNode(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceMemoryBase> operands) override;

  absl::Status UpdateDnnGraphNode(dnn::DnnGraph&, Stream&,
                                  absl::Span<DeviceMemoryBase> operands,
                                  GraphNodeHandle) override;

  absl::StatusOr<GraphNodeHandle> CreateChildNode(
      absl::Span<const GraphNodeHandle> dependencies,
      const CommandBuffer& nested) override;

  absl::Status UpdateChildNode(GraphNodeHandle node_handle,
                               const CommandBuffer& nested) override;

  absl::StatusOr<GraphNodeHandle> CreateKernelNode(
      absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) override;

  absl::Status UpdateKernelNode(GraphNodeHandle node_handle,
                                const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgsPackedArrayBase& args) override;

  absl::StatusOr<GraphNodeHandle> CreateEmptyNode(
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  absl::Status LaunchGraph(Stream* stream) override;

  absl::StatusOr<size_t> GetNodeCount() const override;

  // Set the nodes inside the command buffer to the target priority, cuda
  // currently only support kernel node's priority.
  absl::Status SetPriority(StreamPriority priority) override;

  absl::Status PrepareFinalization() override;

  absl::StatusOr<GraphConditionalHandle> CreateConditionalHandle() override;

  absl::Status WriteGraphToDotFile(absl::string_view path) override;

  absl::Status InstantiateGraph() override;

  using ScopedCudaGraphExec = ScopedGraphExec<CUgraphExec>;
  std::unique_ptr<ScopedUpdateMode> ActivateUpdateMode(
      GpuCommandBuffer* nested_cmd_buffer) override;

  absl::Status CheckCanBeUpdated() override;

  // A signature of a device kernels updating conditional handle(s).
  using SetCaseConditionKernel =
      TypedKernel<CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  DeviceMemory<uint8_t>, bool, int32_t, int32_t, bool>;

  using SetWhileConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<bool>>;

  // Lazy loaded auxiliary kernels required for building CUDA graphs (no-op
  // barriers, updating conditional handles, etc.).
  NoOpKernel noop_kernel_;
  SetCaseConditionKernel set_case_condition_kernel_;
  SetWhileConditionKernel set_while_condition_kernel_;

  StreamExecutor* parent_;

  CudaContext* cuda_context_;

  static_assert(std::is_pointer_v<CUgraph>, "CUgraph must be a pointer");
  static_assert(std::is_pointer_v<CUgraphExec>,
                "CUgraphExec must be a pointer");

  CUgraph graph_ = nullptr;     // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;  // ownership of `graph_`

  CUgraphExec exec_ = nullptr;       // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;  // ownership of `is_owned_graph_exec_`
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
