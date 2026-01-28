#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <iostream>

namespace core {
namespace loader {

    
    struct TensorInfo {
        std::vector<int64_t> shape; 
        std::string dtype;          
        size_t data_offset;         
        size_t data_size;           
    };

    
    class SafeTensorsLoader {
    public:
        
        bool load_header(const std::string& filepath);

        
        TensorInfo get_tensor_info(const std::string& name) const;

        
        void debug_print_info() const;

    private:
        std::unordered_map<std::string, TensorInfo> metadata_;
        size_t header_size_ = 0;
    };

} 
} 