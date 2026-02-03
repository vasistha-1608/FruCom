#include "core/loader/safetensors.h"
#include "core/tensor.h"
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::cout << "FruCom Engine: Initializing Loader..." << std::endl;

    core::loader::SafeTensorsLoader loader;
    
    
    std::string model_path = "../models/model-00001-of-00002.safetensors";
    
  
    if (!fs::exists(model_path)) {
        std::cerr << "CRITICAL ERROR: Model file not found at: " << model_path << std::endl;
        std::cerr << "Please ensure 'models' folder is in the project root." << std::endl;
        return -1;
    }

    if (loader.load_header(model_path)) {
       std::cout << "Success! File Mapped." << std::endl;

        
        std::string test_tensor = "model.embed_tokens.weight";
        void* data = loader.get_tensor_data(test_tensor);

        if (data) {
            uint16_t* weights = static_cast<uint16_t*>(data);
            std::cout << "First 5 weights of " << test_tensor << ": ";
            for(int i=0; i<5; i++) {
                std::cout << weights[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Could not find tensor: " << test_tensor << std::endl;
            loader.debug_print_info(); 
        }
    } else {
        std::cerr << "Failed to load header." << std::endl;
        return -1;
    }

    std::string tensor_name = "model.embed_tokens.weight";
    
    auto info = loader.get_tensor_info(tensor_name);
    void* cpu_data = loader.get_tensor_data(tensor_name);

    if (cpu_data) {
        std::cout << "Found '" << tensor_name << "' on CPU." << std::endl;
        std::cout << "Shape: [" << info.shape[0] << ", " << info.shape[1] << "]" << std::endl;

        
        core::Tensor gpu_tensor(info.shape, tensor_name);

        
        std::cout << "Uploading to GPU..." << std::endl;
        gpu_tensor.to_device(cpu_data, info.data_size);

        
        std::vector<uint16_t> verification_buffer(5); 
        gpu_tensor.to_host(verification_buffer.data(), 5 * sizeof(uint16_t));

        std::cout << "GPU Round-Trip Verified! First 5 values: ";
        for (auto v : verification_buffer) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        std::cout << "SUCCESS: Data successfully moved to VRAM." << std::endl;

    } else {
        std::cerr << "Tensor not found! Did you download the right model?" << std::endl;
    }

    return 0;
}