# Fine-Tuning-LLM
This repository contains a Python implementation for fine-tuning a Llama language model using the LoRA (Low-Rank Adaptation) technique and 4-bit quantization for memory efficiency. The fine-tuned model is trained on a medical terms dataset and can be used for generating context-aware text responses.

# Features
- 4-Bit Quantization: Utilize 4-bit precision for loading and running the Llama model, significantly reducing memory usage while maintaining performance.
- LoRA Fine-Tuning: Apply the LoRA method to introduce additional low-rank layers to the model, enabling parameter-efficient fine-tuning.
- Custom Training Pipeline: Built with Hugging Face's transformers and trl libraries, providing a streamlined process for training and inference.
- Medical Dataset: Fine-tune the model on a dataset of medical terms, making it well-suited for applications in the healthcare domain.
- Text Generation: Leverage the fine-tuned model for generating coherent and contextually relevant text based on user prompts.


# Model Details
- Model: Llama2
- Dataset: Wiki Medical Terms
- Quantization: 4-bit precision with NF4 quantization type.
- LoRA Configuration:
- Rank (r): 64
- LoRA Alpha: 16
- LoRA Dropout: 0.1


You can install the necessary packages directly:

pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
pip install huggingface_hub
