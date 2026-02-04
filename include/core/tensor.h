#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace core {

    enum class DType {
        FLOAT32, // 4 bytes (Standard C++ float)
        FLOAT16  // 2 bytes (The Model Weights)
    };

    class Tensor {
    public:
        
        Tensor(const std::vector<int64_t>& shape, DType dtype = DType::FLOAT32, const std::string& name = "params");
        
        
        ~Tensor();

        
        void to_device(const void* host_data, size_t size_in_bytes);

       
        void to_host(void* host_buffer, size_t size_in_bytes) const;

        DType dtype() const {return dtype_;}
        void* device_data() const { return d_data_; }
        size_t size() const { return size_in_bytes_; }
        std::vector<int64_t> shape() const { return shape_; }

    private:
        std::string name_;
        std::vector<int64_t> shape_;
        size_t size_in_bytes_ = 0;
        DType dtype_;
        
        void* d_data_ = nullptr;
    };

} 