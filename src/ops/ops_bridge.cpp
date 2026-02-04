#include "core/ops/ops.h"
#include <stdexcept>


extern "C" void launch_rms_norm_fp16(void* out, const void* in, const void* w, int rows, int cols);

namespace core {
namespace ops {

    void rms_norm(Tensor& output, const Tensor& input, const Tensor& weight) {
        
        if (input.shape() != output.shape()) {
            throw std::runtime_error("RMSNorm Error: Input and Output shapes mismatch");
        }

        
        int rows = input.shape()[0];
        int cols = input.shape()[1];

        
        if (input.dtype() == DType::FLOAT16) {
            // Launch the new FP16 kernel
            launch_rms_norm_fp16(
                output.device_data(),
                input.device_data(),
                weight.device_data(),
                rows,
                cols
            );
        } 
        else {
            
            std::cerr << "Error: Only FP16 tensors are supported for now!" << std::endl;
        }
    
    }

} 
} 