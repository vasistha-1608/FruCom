#include "core/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace core {

    
    size_t get_element_count(const std::vector<int64_t>& shape) {
        size_t count = 1;
        for (auto dim : shape) count *= dim;
        return count;
    }

    Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype,const std::string& name) 
        : shape_(shape),dtype_(dtype) , name_(name) {
        
    
       
        size_t num_elements = get_element_count(shape);
      size_t element_size = 4; // Default FP32
        if (dtype == DType::FLOAT16) {
            element_size = 2;
        }

        size_in_bytes_ = num_elements * element_size; 

        

        
        cudaError_t err = cudaMalloc(&d_data_, size_in_bytes_);
        
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (Malloc): " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to allocate GPU memory");
        }
        
        std::cout << "[GPU Alloc] " << name_ << ": " 
                  << (size_in_bytes_ / 1024.0 / 1024.0) << " MB" << std::endl;
    }

    Tensor::~Tensor() {
        if (d_data_) {
            cudaFree(d_data_);
        }
    }

    void Tensor::to_device(const void* host_data, size_t size_in_bytes) {
        if (size_in_bytes != size_in_bytes_) {
            std::cerr << "Warning: Size mismatch in upload! Expected " << size_in_bytes_ 
                      << " got " << size_in_bytes << std::endl;
        }

        
        cudaError_t err = cudaMemcpy(d_data_, host_data, size_in_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (Memcpy H2D): " << cudaGetErrorString(err) << std::endl;
        }
    }

    void Tensor::to_host(void* host_buffer, size_t size_in_bytes) const {
        
        cudaError_t err = cudaMemcpy(host_buffer, d_data_, size_in_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (Memcpy D2H): " << cudaGetErrorString(err) << std::endl;
        }
    }

} 