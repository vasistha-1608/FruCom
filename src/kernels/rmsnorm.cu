#include <cuda_runtime.h>
#include<cuda_fp16.h>
#include <cmath>


extern "C" {

    
    __global__ void rms_norm_kernel_fp16(
        half* output,          
        const half* input,    
        const half* weight,    
        int hidden_dim,        
        int num_tokens,         
        float epsilon           
    ) {
       
        int row = blockIdx.x;
        if (row >= num_tokens) return;

        
        const half* row_input = input + row * hidden_dim;
        half* row_output = output + row * hidden_dim;
        //  Optimize this with parallel reduction later!
        if (threadIdx.x == 0) {
            float sum_sq = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                float val = __half2float(row_input[i]);
                sum_sq += val * val;
            }

           
            float rms = rsqrtf((sum_sq / hidden_dim) + epsilon);

            
            for (int i = 0; i < hidden_dim; i++) {
                float val = __half2float(row_input[i]);
                float w   = __half2float(weight[i]);
                
                
                row_output[i] = __float2half(val * rms * w);
            }
        }
    }

    
    // void launch_rms_norm(
    //     float* output, 
    //     const float* input, 
    //     const float* weight, 
    //     int rows, 
    //     int cols
    // ) {
        
    //     dim3 grid(rows);
    //     dim3 block(1); 

    //     rms_norm_kernel<<<grid, block>>>(output, input, weight, cols, rows, 1e-5f);

    void launch_rms_norm_fp16(void* out, const void* in, const void* w, int rows, int cols) {
        dim3 grid(rows);
        dim3 block(1); 
        
        rms_norm_kernel_fp16<<<grid, block>>>(
            (half*)out, 
            (half*)in, 
            (half*)w, 
            cols, 
            rows, 
            1e-5f
        );
    }
}