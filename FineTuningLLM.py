import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

# Load a pre-trained Llama model from Hugging Face's model hub, with quantization applied to reduce memory usage.
# The model is configured to load in 4-bit precision for reduced memory footprint, 
# with the computation performed in 16-bit floating-point precision (float16).
llama_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",  # The model's name or path
    quantization_config=BitsAndBytesConfig(  # Configuration for the 4-bit quantization
        load_in_4bit=True,  # Load model weights in 4-bit precision
        bnb_4bit_compute_dtype=getattr(torch, "float16"),  # Use float16 for computation
        bnb_4bit_quant_type="nf4"  # Quantization type, where "nf4" is one of the supported types
    )
)

# Modify the model configuration to not use the cache during training and inference.
# The cache is typically used for speeding up inference by reusing previously computed key-value pairs.
llama_model.config.use_cache = False  # Disabling the cache

# Set the model's pretraining tensor parallelism to 1, which might be a custom parameter specific to this model.
llama_model.config.pretrainin_tp = 1  # Custom configuration for tensor parallelism

# Load the corresponding tokenizer for the pre-trained Llama model.
# The tokenizer is responsible for converting text into tokens that the model can understand.
llama_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",  # The model's tokenizer name or path
    trust_remote_code=True  # Allow loading of custom or non-standard tokenizer code from the model hub
)

# Set the padding token to be the same as the end-of-sequence token (EOS).
# This is necessary when the tokenizer needs to pad sequences to a uniform length.
llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Set padding token

# Specify that padding should occur on the right side of the sequence.
# This means that shorter sequences will be padded on the right to match the length of the longest sequence in the batch.
llama_tokenizer.padding_side = "right"  # Pad sequences on the right

# Define the arguments for training the model, including where to save results, batch size, and the maximum number of training steps.
training_arguments = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints and training results
    per_device_train_batch_size=4,  # Number of training samples per device in a batch
    max_steps=100  # Total number of training steps to perform
)

# Initialize the Supervised Fine-Tuning (SFT) trainer, which orchestrates the training process.
# The SFT trainer is set up with the model, training arguments, dataset, tokenizer, and a PEFT (Parameter-Efficient Fine-Tuning) configuration.
llama_sft_trainer = SFTTrainer(
    model=llama_model,  # The pre-trained model to fine-tune
    args=training_arguments,  # Training arguments
    train_dataset=load_dataset(  # Load the dataset to be used for fine-tuning
        path="aboonaji/wiki_medical_terms_llam2_format",  # Dataset path or identifier
        split="train"  # Use the training split of the dataset
    ),
    tokenizer=llama_tokenizer,  # Tokenizer to process the text data
    peft_config=LoraConfig(  # Configuration for LoRA, a PEFT technique that adds small, learnable weights to the model
        task_type="CAUSAL_LM",  # Type of task, here "causal language modeling"
        r=64,  # Rank of the low-rank matrices in LoRA
        lora_alpha=16,  # Scaling factor for LoRA updates
        lora_dropout=0.1  # Dropout rate to prevent overfitting in LoRA
    ),
    dataset_text_field="text"  # The field in the dataset that contains the text data
)

# Start the training process using the SFT trainer.
# The model will be fine-tuned on the provided dataset according to the specified training arguments.
llama_sft_trainer.train()

# Define a user prompt that will be fed to the model for text generation after fine-tuning.
user_prompt = "What is Chest Pain?"

# Create a text generation pipeline that leverages the fine-tuned model and tokenizer.
# The pipeline simplifies the process of generating text, handling tokenization and decoding.
text_generation_pipeline = pipeline(
    task="text-generation",  # The type of task for the pipeline, in this case, generating text
    model=llama_model,  # The fine-tuned model to use for generation
    tokenizer=llama_tokenizer,  # The tokenizer to process input and decode output
    max_length=400  # Maximum length of the generated text
)

# Generate a response from the model using the text generation pipeline.
# The user prompt is wrapped with special tokens (e.g., <s>[INST]) that may be model-specific.
model_answer = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")

# Print the entire output of the model, which is a list of generated sequences.
print(model_answer)

# Print only the generated text from the first sequence in the output.
print(model_answer[0]['generated_text'])
