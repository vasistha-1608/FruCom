#pragma once
#include "core/tensor.h"

namespace core {
namespace ops {

    
    void rms_norm(Tensor& output, const Tensor& input, const Tensor& weight);

} 
} 