#include "core/loader/safetensors.h"
#include "core/common/json.hpp" 
#include <fstream>
#include <iostream>


using json = nlohmann::json;

namespace core {
namespace loader {

    bool SafeTensorsLoader::load_header(const std::string& filepath) {
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return false;
        }

        
        uint64_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));

        if (header_len <= 0) {
            std::cerr << "Error: Invalid header length " << header_len << std::endl;
            return false;
        }

        
        std::string header_str;
        header_str.resize(header_len);
        file.read(&header_str[0], header_len);

       
        try {
            auto j = json::parse(header_str);

            
            for (auto& [key, value] : j.items()) {
                
                if (key == "__metadata__") continue;

                TensorInfo info;
                info.dtype = value["dtype"].get<std::string>();
                
                
                size_t start = value["data_offsets"][0];
                size_t end = value["data_offsets"][1];
                
                
                info.data_offset = 8 + header_len + start;
                info.data_size = end - start;

                
                for (auto& dim : value["shape"]) {
                    info.shape.push_back(dim);
                }

                
                metadata_[key] = info;
            }
            
            header_size_ = header_len;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "JSON Parsing Error: " << e.what() << std::endl;
            return false;
        }
    }

    TensorInfo SafeTensorsLoader::get_tensor_info(const std::string& name) const {
        if (metadata_.find(name) != metadata_.end()) {
            return metadata_.at(name);
        }
        return {}; 
    }

    void SafeTensorsLoader::debug_print_info() const {
        std::cout << "=== Loaded " << metadata_.size() << " Tensors ===" << std::endl;
        
        
        int count = 0;
        for (const auto& [name, info] : metadata_) {
            if (count++ > 5) break;
            
            std::cout << "Tensor: " << name << " | Shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                std::cout << info.shape[i] << (i < info.shape.size() - 1 ? ", " : "");
            }
            std::cout << "] | Type: " << info.dtype << std::endl;
        }
        std::cout << "..." << std::endl;
    }

} 
} 