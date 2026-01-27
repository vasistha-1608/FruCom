# [FruCom(Frugal Compute)]

> **A high-performance, memory-efficient inference engine for Small Language Models (SLMs) on consumer-grade NVIDIA GPUs.**

## 📌 Project Overview
**[FruCom]** is a custom-built C++/CUDA library designed to solve a specific problem: running modern industry-standard Small Language Models (SLMs) on hardware with strict memory constraints (6GB – 8GB VRAM).

Current inference engines often prioritize data center hardware (A100/H100) or rely on heavy frameworks like PyTorch. This project aims to strip away the abstraction layers, communicating directly with the GPU to maximize token throughput and minimize memory footprint.

## 🎯 Goals & Philosophy
* **Memory is King:** The primary constraint is fitting useful models (e.g., Phi-3, Gemma, Llama-3-8B) into limited VRAM without crashing.
* **Speed Matters:** Inference must be interactive. I aim for low-latency token generation suitable for real-time chat.
* **Zero-Dependency Core:** The core runtime is built on raw C++ and CUDA. I avoided heavy dependencies like Torch or TensorFlow in the production build.
* **Educational & Experimental:** A "from scratch" approach to understanding the systems engineering behind LLM inference.

## 🛠️ Tech Stack
* **Language:** C++ (Host logic), CUDA (Device kernels)
* **Build System:** CMake
* **Target Hardware:** NVIDIA GPUs (Ampere/Ada Lovelace architectures preferred) with 6GB+ VRAM.
* **Supported OS:** Linux & Windows

## 👥 Target Audience
* **Students & Researchers:** Who want to run experiments on local gaming laptops (e.g., RTX 3060/4050/4060) without cloud costs.
* **Edge Developers:** Deploying AI on resource-constrained devices or desktops where the GPU is shared.
* **Systems Engineers:** Developers looking to understand the low-level implementation of Transformer inference.

## 🗺️ Roadmap
### Phase 1: Foundations (Current Focus)
- [ ] Define project architecture and requirements.
- [ ] Implement Safetensors model loader (C++).
- [ ] Basic memory management and VRAM allocation strategies.

### Phase 2: The Core
- [ ] Implement FP16 matrix multiplication kernels in CUDA.
- [ ] Implement Transformer blocks (Attention, FeedForward, RMSNorm).
- [ ] Validating inference on "Tiny" models (e.g., Phi-3 Mini).

### Phase 3: Optimization & Scale
- [ ] KV Cache management for longer context windows.
- [ ] Integration of Python/Triton bindings for rapid prototyping.
- [ ] Support for 4-bit Quantization (to fit 7B/8B models on <8GB VRAM).

## ⚠️ Scope & Limitations
* **Inference Only:** This library does not support model training or fine-tuning.
* **Single GPU:** Optimized for single-card setups; multi-GPU distributed inference is out of scope.
* **NVIDIA Only:** Current support is strictly for CUDA-enabled devices.

## 🤝 Contributing
This project is currently in the **Pre-Alpha / Architecture** phase. We are laying the groundwork before writing the core kernels.
