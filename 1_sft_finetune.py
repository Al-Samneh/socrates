# 1_sft_finetune.py
# Stage 1: Initial Supervised Fine-Tuning with Stability Fixes

print("Starting Stage 1: Stable Supervised Fine-Tuning...")

# --- 1. Imports ---
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


from huggingface_hub import login
import getpass

# Securely prompt the user to enter their Hugging Face token
hf_token = getpass.getpass("Enter your Hugging Face token: ")

# Login to Hugging Face using the provided token
login(token=hf_token)

print("Successfully logged in to Hugging Face!")




# --- 2. Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
SFT_ADAPTER_OUTPUT_DIR = "./socrates-sft-adapters"   # We will save the LoRA "adapters"




# Configure 4-bit quantization to reduce memory and stabilize training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --- 3. Data Loading and Formatting ---
print("\nStep 3: Loading and formatting dataset...")
dataset = load_dataset("tylercross/platos_socrates_no_context", split='train')

# Function to clean speaker names from text
import re

def clean_speaker_names(text):
    """Remove speaker names like 'SOCRATES:', 'MENO:', etc. from the beginning of text"""
    # Remove patterns like "SPEAKER_NAME:" at the start of text
    cleaned = re.sub(r'^[A-Z]{2,}:\s*', '', text.strip())
    return cleaned

# Use a standard instruction-following format for stability and better performance
def format_prompt(example):
    # Clean both input and output of speaker names
    clean_input = clean_speaker_names(example['input'])
    clean_output = clean_speaker_names(example['output'])
    text = f"<s>[INST] {clean_input} [/INST] {clean_output}"
    return {"text": text}

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

formatted_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
print(f"Sample formatted entry:\n{formatted_dataset[0]['text']}")

# --- 4. Model and Tokenizer Setup ---
print("\nStep 4: Loading Model, Tokenizer, and applying LoRA...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME) # Huggingface tokenizer
tokenizer.pad_token = tokenizer.eos_token # to let the model know the padding token
tokenizer.padding_side = "right" # Important for avoiding issues with some model types

# Tokenize the dataset
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config, # 4-bit quantization
    device_map="auto" # to let the model know the device map
)

model.config.use_cache = False # Disable during training
model = prepare_model_for_kbit_training(model) # to let the model know the model for kbit training

# Configure LoRA for efficient training
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 5. Training ---
print("\nStep 5: Configuring and starting training...")
training_args = TrainingArguments(
    output_dir="./socrates-training-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit", # Memory-efficient optimizer
    num_train_epochs=3,
    save_steps=200,
    logging_steps=25,
    learning_rate=2e-5, # Lowered learning rate for stability
    weight_decay=0.001,
    bf16=True, # Use bfloat16 for H100
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    data_collator=data_collator,
)
trainer.train()

# --- 6. Save ---
print(f"\nStep 6: Saving the fine-tuned LoRA adapters to {SFT_ADAPTER_OUTPUT_DIR}...")
trainer.model.save_pretrained(SFT_ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(SFT_ADAPTER_OUTPUT_DIR)

print("\n✅ Stage 1 Complete. The foundational Socrates model adapters are saved.")