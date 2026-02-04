#include "core/loader/safetensors.h"
#include "core/common/json.hpp" 
#include <fstream>
#include <iostream>


#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

using json = nlohmann::json;

namespace core {
namespace loader {

    SafeTensorsLoader::~SafeTensorsLoader() {
#ifdef _WIN32
        if (mapped_ptr_) UnmapViewOfFile(mapped_ptr_);
        if (h_map_) CloseHandle(h_map_);
        if (h_file_ && h_file_ != INVALID_HANDLE_VALUE) CloseHandle(h_file_);
#else
        if (mapped_ptr_ && mapped_ptr_ != MAP_FAILED) munmap(mapped_ptr_, file_size_);
        if (fd_ != -1) close(fd_);
#endif
    }

    bool SafeTensorsLoader::load_header(const std::string& filepath) {
#ifdef _WIN32
        
        h_file_ = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h_file_ == INVALID_HANDLE_VALUE) {
            std::cerr << "Windows Error: Could not open file." << std::endl;
            return false;
        }

        
        h_map_ = CreateFileMappingA(h_file_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (h_map_ == NULL) {
            std::cerr << "Windows Error: Could not create file mapping." << std::endl;
            CloseHandle(h_file_);
            return false;
        }

        
        mapped_ptr_ = MapViewOfFile(h_map_, FILE_MAP_READ, 0, 0, 0);
        if (mapped_ptr_ == NULL) {
            std::cerr << "Windows Error: Could not map view of file." << std::endl;
            CloseHandle(h_map_);
            CloseHandle(h_file_);
            return false;
            
        }
#else
        // LINUX Implementation (Standard mmap)
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) return false;

        struct stat sb;
        if (fstat(fd_, &sb) == -1) { close(fd_); return false; }
        file_size_ = sb.st_size;

        mapped_ptr_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_ptr_ == MAP_FAILED) { close(fd_); return false; }
#endif

        
        char* raw_ptr = static_cast<char*>(mapped_ptr_);
        uint64_t header_len = *reinterpret_cast<uint64_t*>(raw_ptr);

        
        std::string header_str(raw_ptr + 8, header_len);

        try {
            auto j = json::parse(header_str);

            for (auto& [key, value] : j.items()) {
                if (key == "__metadata__") continue;

                TensorInfo info;
                info.dtype = value["dtype"].get<std::string>();
                size_t start = value["data_offsets"][0];
                size_t end = value["data_offsets"][1];

                
                size_t global_offset = 8 + header_len + start;

                info.data_offset = global_offset;
                info.data_size = end - start;
                
                for (auto& dim : value["shape"]) info.shape.push_back(dim);

                metadata_[key] = info;
            }
            header_size_ = header_len;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "JSON Error: " << e.what() << std::endl;
            return false;
        }
    
    }

    TensorInfo SafeTensorsLoader::get_tensor_info(const std::string& name) const {
        if (metadata_.find(name) != metadata_.end()) {
            return metadata_.at(name);
        }
        return {}; 
    }

    void* SafeTensorsLoader::get_tensor_data(const std::string& name)  {
        if (metadata_.find(name) == metadata_.end()) return nullptr;
        
        // Calculate the exact address in RAM
        TensorInfo& info = metadata_[name];
        char* base = static_cast<char*>(mapped_ptr_);
        return static_cast<void*>(base + info.data_offset);
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