#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/tests/llvm_irgen_test_base.h"

namespace xla {
namespace gpu {

class RocmRnnBackendTest : public LlvmIrGenTestBase {};

// Test that an RNN with seq_lengths is lowered to the correct FFI custom call.
TEST_F(RocmRnnBackendTest, RnnWithSeqLengthsEmitsFfiCall) {
  const std::string hlo_text = R"(
HloModule LstmWithPadding

ENTRY %main {
  %input = f32[5,2,3] parameter(0)
  %h_0 = f32[1,2,4] parameter(1)
  %c_0 = f32[1,2,4] parameter(2)
  %weights = f32[1,7,16] parameter(3)
  %seq_lengths = s32[2] parameter(4)

  ROOT custom-call = (f32[5,2,4], f32[1,2,4], f32[1,2,4]) 
       custom-call(%input, %h_0, %c_0, %weights, %seq_lengths),
       custom_call_target="RnnForwardWithSeqLengths",
       backend_config={kind = "lstm", is_training = false},
       api_version=API_VERSION_TYPED_FFI
}
)";

  // Check that the IR contains a call to the FFI handler.
  CompileAndVerifyIr(hlo_text, R"(
    // CHECK: call @xla.gpu.ffi.handler
    // CHECK: "RnnForwardWithSeqLengths"
  )",
                     /*match_optimized_hlo=*/false);
}

// Test that a standard RNN (without seq_lengths) is also lowered correctly.
TEST_F(RocmRnnBackendTest, RnnWithoutSeqLengthsEmitsFfiCall) {
  const char* const hlo_text = R"(
HloModule LstmWithoutPadding

ENTRY main {
  %input = f32[5,2,3] parameter(0)
  %h_0 = f32[1,2,4] parameter(1)
  %c_0 = f32[1,2,4] parameter(2)
  %weights = f32[1,7,16] parameter(3)

  ROOT %rnn = (f32[5,2,4], f32[1,2,4], f32[1,2,4]) custom-call(
      %input, %h_0, %c_0, %weights),
      custom_call_target="RnnForward",
      backend_config={kind = "lstm", is_training = false},
      api_version=API_VERSION_TYPED_FFI
}
)";

  CompileAndVerifyIr(hlo_text, R"(
    // CHECK: call @xla.gpu.ffi.handler
    // CHECK: "RnnForward"
  )",
                     /*match_optimized_hlo=*/false);
}

}  // namespace gpu
}  // namespace xla
