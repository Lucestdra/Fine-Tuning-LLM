# Fine-Tuning-LLM
This repository contains a Python implementation for fine-tuning a Llama language model using the LoRA (Low-Rank Adaptation) technique and 4-bit quantization for memory efficiency. The fine-tuned model is trained on a medical terms dataset and can be used for generating context-aware text responses.

# Features
- 4-Bit Quantization: Utilize 4-bit precision for loading and running the Llama model, significantly reducing memory usage while maintaining performance.
- LoRA Fine-Tuning: Apply the LoRA method to introduce additional low-rank layers to the model, enabling parameter-efficient fine-tuning.
- Custom Training Pipeline: Built with Hugging Face's transformers and trl libraries, providing a streamlined process for training and inference.
- Medical Dataset: Fine-tune the model on a dataset of medical terms, making it well-suited for applications in the healthcare domain.
- Text Generation: Leverage the fine-tuned model for generating coherent and contextually relevant text based on user prompts.
