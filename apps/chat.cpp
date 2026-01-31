#include "core/loader/safetensors.h"
#include <iostream>
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

    return 0;
}