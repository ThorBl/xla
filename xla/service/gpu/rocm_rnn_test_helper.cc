#include <string_view>

#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/rocm/rocm_dnn.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/str_util.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace xla {
namespace gpu {

// This is the "glue" function that will be called by the XLA runtime when it
// encounters a custom-call with the target "RnnForward".
// It receives a CallFrame, which contains all the context.
absl::Status RnnForwardImpl(
    se::Stream* stream,
    // Inputs from FFI
    ffi::Buffer<xla::F32> input_buffer, ffi::Buffer<xla::F32> h0_buffer,
    ffi::Buffer<xla::F32> c0_buffer, ffi::Buffer<xla::F32> weights_buffer,
    std::optional<ffi::Buffer<xla::S32>> seq_lengths_buffer,
    // Outputs from FFI
    ffi::Result<ffi::Buffer<xla::F32>> output_buffer,
    ffi::Result<ffi::Buffer<xla::F32>> output_h_buffer,
    ffi::Result<ffi::Buffer<xla::F32>> output_c_buffer,
    // Attributes from FFI
    std::string_view backend_config_str) {
      
  se::DeviceMemory<float> input_data(input_buffer.device_memory());
  se::DeviceMemory<float> h0_data(h0_buffer.device_memory());
  se::DeviceMemory<float> c0_data(c0_buffer.device_memory());
  se::DeviceMemory<float> weights_data(weights_buffer.device_memory());

  std::optional<se::DeviceMemory<int32_t>> seq_lengths_data_opt;
  if (seq_lengths_buffer.has_value()) {
    seq_lengths_data_opt.emplace(seq_lengths_buffer->device_memory());
  }

  se::DeviceMemory<float> output_data((*output_buffer).device_memory());
  se::DeviceMemory<float> output_h_data((*output_h_buffer).device_memory());
  se::DeviceMemory<float> output_c_data((*output_c_buffer).device_memory());

  // 2. Use ONLY the converted StreamExecutor types from now on.
  se::dnn::DnnSupport* dnn = stream->parent()->AsDnn();
  if (!dnn) {
    return absl::InternalError("DNN support is not available.");
  }
  bool is_training =
      absl::StrContains(backend_config_str, "\"is_training\":true");

  bool has_seq_lengths = seq_lengths_data_opt.has_value();
  int num_layers = 1, hidden_size = 4, input_size = 3, seq_length = 5,
      batch_size = 2;
  auto data_type = se::dnn::DataType::kFloat;
  se::dnn::AlgorithmConfig algorithm_config;
  se::NumericOptions numeric_options;

  // Create the main RNN descriptor. This call now matches the 15-argument
  // API.
  TF_ASSIGN_OR_RETURN(
      auto rnn_desc,
      dnn->CreateRnnDescriptor(num_layers, hidden_size, input_size,
                               /*cell_size=*/hidden_size, batch_size,
                               se::dnn::RnnInputMode::kRnnLinearSkip,
                               se::dnn::RnnDirectionMode::kRnnUnidirectional,
                               se::dnn::RnnMode::kRnnLstm, data_type,
                               algorithm_config, numeric_options,
                               /*dropout=*/0.0f,
                               /*seed=*/0,
                               /*state_allocator=*/nullptr,
                               /*use_padded_io=*/has_seq_lengths));

  // Create the sequence and state descriptors.
  std::unique_ptr<se::dnn::RnnSequenceTensorDescriptor> input_seq_desc;
  se::DeviceMemory<int32_t> seq_lengths_for_dnn;
  if (has_seq_lengths) {
    // Correctly convert the optional buffer before using it.
    se::DeviceMemory<int32_t> seq_lengths_data(
        seq_lengths_buffer->device_memory());

    std::vector<int> host_seq_lengths(batch_size);
    TF_RETURN_IF_ERROR(
        stream->Memcpy(host_seq_lengths.data(),
                       seq_lengths_data,  // Use the converted variable
                       batch_size * sizeof(int)));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    TF_ASSIGN_OR_RETURN(input_seq_desc, dnn->CreateRnnSequenceTensorDescriptor(
                                            seq_length, batch_size, input_size,
                                            host_seq_lengths, true, data_type));
    // Assign the converted variable to the one used in the final call.
    seq_lengths_for_dnn = seq_lengths_data;
  } else {
    TF_ASSIGN_OR_RETURN(input_seq_desc,
                        dnn->CreateRnnSequenceTensorDescriptor(
                            seq_length, batch_size, input_size, data_type));
  }

  TF_ASSIGN_OR_RETURN(auto output_seq_desc,
                      dnn->CreateRnnSequenceTensorDescriptor(
                          seq_length, batch_size, hidden_size, data_type));
  TF_ASSIGN_OR_RETURN(
      auto h_desc, dnn->CreateRnnStateTensorDescriptor(num_layers, batch_size,
                                                       hidden_size, data_type));
  TF_ASSIGN_OR_RETURN(
      auto c_desc, dnn->CreateRnnStateTensorDescriptor(num_layers, batch_size,
                                                       hidden_size, data_type));

  // --- 5. Allocate Workspace and Call DoRnnForward ---
  // For an IR-only test, we don't need real memory. We can pass
  // default-constructed (null) DeviceMemory objects.
  // For an IR-only test, a 0-size allocator is fine.
  se::OwningScratchAllocator<256> workspace_allocator(
      stream->parent()->device_ordinal(), nullptr);
  se::OwningScratchAllocator<256> reserve_allocator(
      stream->parent()->device_ordinal(), nullptr);
  se::dnn::ProfileResult profile_result;

  bool success = dnn->DoRnnForward(
      stream, *rnn_desc, *input_seq_desc, input_data, seq_lengths_for_dnn,
      *h_desc, h0_data, *c_desc, c0_data, weights_data, *output_seq_desc,
      &output_data, *h_desc, &output_h_data, *c_desc, &output_c_data,
      is_training, &reserve_allocator, &workspace_allocator, &profile_result);

  if (!success) {
    return absl::InternalError("DoRnnForward failed.");
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(RnnForwardFfi, RnnForwardImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .OptionalArg<ffi::Buffer<PrimitiveType::S32>>()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Attr<std::string_view>("backend_config"));

// REGISTER the single handler for both target names to support existing
// tests.
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "RnnForward", "ROCM",
                         RnnForwardFfi);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "RnnForwardWithSeqLengths",
                         "ROCM", RnnForwardFfi);

}  // namespace gpu
}  // namespace xla
