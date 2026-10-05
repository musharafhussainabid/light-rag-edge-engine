# Edge AI Execution & Small Language Model Quantization

## Overview
Deploying Large Language Models (LLMs) on resource-constrained edge hardware presents severe challenges regarding memory footprint (RAM/VRAM) and inference latency. Small Language Models (SLMs)—ranging from 1B to 3B parameters—combined with post-training quantization (PTQ) techniques enable high-throughput, sub-second natural language processing directly on CPU devices.

## Quantization Formats & Memory Savings
Model quantization reduces the precision of model weights from standard floating-point formats (FP16 or FP32) to low-bit integer representations (INT8, INT4, or GGUF variants).

1. **FP16 (Half Precision)**:
   - Memory required: ~2 GB per 1B parameters.
   - Requires high memory bandwidth and specialized hardware acceleration for real-time throughput.

2. **GGUF INT4 (4-bit Quantization)**:
   - Memory required: ~0.6 GB - 0.8 GB per 1B parameters.
   - Reduces RAM usage by roughly 70-75% compared to FP16.
   - Maintains over 95% of downstream task accuracy while drastically reducing Time-To-First-Token (TTFT).

3. **GGUF INT8 (8-bit Quantization)**:
   - Memory required: ~1.1 GB per 1B parameters.
   - Serves as a balanced midpoint between baseline precision and extreme compression.

## Retrieval-Augmented Generation (RAG) on Edge Devices
Standard vector retrieval relies on dense numerical representations. In on-device Light-RAG architectures:
- **Embedding Generation**: Small transformer encoders (such as `all-MiniLM-L6-v2`) project document chunks into 384-dimensional vector spaces.
- **Vector Search**: Indexing frameworks like FAISS execute flat L2 or Inner Product vector comparisons in CPU L3 cache within milliseconds.
- **Context Injection**: Top-$k$ relevant chunks are dynamically retrieved and injected into the system prompt of the quantized SLM engine (`llama-cpp-python` / `ONNX Runtime`), ensuring low latency and strictly deterministic source grounding.

## Performance Metrics & Goals
Target benchmark performance on standard consumer CPU configurations (Intel i7 / Apple Silicon):
- **RAM Overhead**: Below 1.2 GB total system footprint.
- **Generation Speed**: Exceeding 15-20 tokens per second.
- **First Token Latency**: Under 400 milliseconds.