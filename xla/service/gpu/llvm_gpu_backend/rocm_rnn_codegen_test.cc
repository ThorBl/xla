#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/tests/llvm_irgen_test_base.h"

namespace xla {
namespace gpu {

class RocmRnnBackendTest : public LlvmIrGenTestBase {};

// Test that an RNN with seq_lengths is lowered to the MIOpen padded sequence
// kernel in the LLVM IR.

TEST_F(RocmRnnBackendTest, RnnWithSeqLengthsEmitsPaddedKernelCall) {
  const char* const hlo_text = R"(
HloModule LstmWithPadding

ENTRY main {
  %input = f32[5,2,3] parameter(0)
  %h_0 = f32[1,2,4] parameter(1)
  %c_0 = f32[1,2,4] parameter(2)
  %weights = f32[1,7,16] parameter(3)
  %seq_lengths = s32[2] parameter(4)

  ROOT %rnn = (f32[5,2,4], f32[1,2,4], f32[1,2,4]) custom-call(
      %input, %h_0, %c_0, %weights, %seq_lengths),
      custom_call_target="RnnForward",
      backend_config="{\"kind\":\"lstm\",\"is_training\":false}"
}
)";

  // Check for the specific MIOpen function call that the new code path
  // generates.
  CompileAndVerifyIr(hlo_text, R"(
    // CHECK: call {{.*}} @miopenRNNForward(
    // CHECK-NOT: call {{.*}} @miopenRNNForwardInference(
  )",
                     /*match_optimized_hlo=*/false);
}


// Test that a standard RNN (without seq_lengths) correctly falls back
// to the default non-padded kernel implementation.
TEST_F(RocmRnnBackendTest, RnnWithoutSeqLengthsEmitsDefaultKernelCall) {
  // This HLO does NOT have `seq_lengths`, so it should use the old path.
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
      backend_config="{\"kind\":\"lstm\",\"is_training\":false}"
}
)";

  // Check for the opposite conditions as in
  // RnnWithSeqLengthsEmitsPaddedKernelCall.
  CompileAndVerifyIr(hlo_text, R"(
    // CHECK: call {{.*}} @miopenRNNForwardInference(
    // CHECK-NOT: call {{.*}} @miopenRNNForward(
  )",
                     /*match_optimized_hlo=*/false);
}

}  // namespace gpu
}  // namespace xla
