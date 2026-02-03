#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace core {

    class Tensor {
    public:
        
        Tensor(const std::vector<int64_t>& shape, const std::string& name = "params");
        
        
        ~Tensor();

        
        void to_device(const void* host_data, size_t size_in_bytes);

       
        void to_host(void* host_buffer, size_t size_in_bytes) const;

       
        void* device_data() const { return d_data_; }
        size_t size() const { return size_in_bytes_; }
        std::vector<int64_t> shape() const { return shape_; }

    private:
        std::string name_;
        std::vector<int64_t> shape_;
        size_t size_in_bytes_ = 0;
        
        
        void* d_data_ = nullptr;
    };

} 