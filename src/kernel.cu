#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = x[index] + y[index];
    }
}

void launch_add(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        out.data_ptr<float>(), 
        n
    );
}