#include "core/loader/safetensors.h"
#include "core/tensor.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include "core/ops/ops.h"

namespace fs = std::filesystem;

int main() {
    std::cout << "FruCom Engine: Initializing Loader..." << std::endl;

    core::loader::SafeTensorsLoader loader;
    
    
    std::string model_path = "../models/model-00001-of-00002.safetensors";
    
  
    if (!loader.load_header(model_path)) return -1;

    std::string weight_name = "model.layers.0.input_layernorm.weight";
    auto info = loader.get_tensor_info(weight_name);
    void* cpu_weights = loader.get_tensor_data(weight_name);

    if (!cpu_weights) {
        std::cerr << "Could not find " << weight_name << std::endl;
        return -1;
    }

    int hidden_dim = info.shape[0]; 
    int batch_size = 1;

    std::cout << "Loaded Weights: " << weight_name << " [" << hidden_dim << "]" << std::endl;

    
    core::Tensor weight_tensor({hidden_dim}, core::DType::FLOAT16, "ln_weights");
    core::Tensor input_tensor({batch_size, hidden_dim}, core::DType::FLOAT16, "input");
    core::Tensor output_tensor({batch_size, hidden_dim}, core::DType::FLOAT16, "output");


    weight_tensor.to_device(cpu_weights, info.data_size);

    std::vector<uint16_t> host_input(hidden_dim, 0x3C00); 
    input_tensor.to_device(host_input.data(), hidden_dim * 2);

 
    std::cout << "Running FP16 RMSNorm..." << std::endl;
    core::ops::rms_norm(output_tensor, input_tensor, weight_tensor);

 
    std::vector<uint16_t> result(5);
    output_tensor.to_host(result.data(), 5 * 2);

    std::cout << "Result (First 5 raw hex values): ";
    for (auto v : result) {
        std::cout << std::hex << v << " ";
    }
    std::cout << std::dec << std::endl;

    return 0;

}