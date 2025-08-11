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
import re
from huggingface_hub import login
import getpass
import os
import json





# Securely prompt the user to enter their Hugging Face token, for safety
hf_token = getpass.getpass("Enter your Hugging Face token: ")

# Login to Hugging Face using the provided token
login(token=hf_token)

print("Successfully logged in to Hugging Face!")




# --- 2. Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1" # base model
SFT_ADAPTER_OUTPUT_DIR = "./socrates-sft-adapters"   # We will save the LoRA "adapters" , this is where the model will be saved




# Configure 4-bit quantization to reduce memory and stabilize training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --- 3. Data Loading and Formatting ---
print("\nStep 3: Loading and formatting dataset...")

def create_merged_dataset():
    """Create merged dataset if it doesn't exist, then load it."""
    merged_file = "data/complete_socratic_training_dataset.json"
    
    # Check if merged dataset already exists
    if os.path.exists(merged_file):
        print(f"Found existing merged dataset: {merged_file}")
    else:
        print("Creating merged dataset from all sources...")
        
        all_entries = []
        
        # 1. Load Hugging Face dataset
        try:
            hf_dataset = load_dataset("tylercross/platos_socrates_no_context", split='train')
            hf_entries = [{"input": item["input"], "output": item["output"]} for item in hf_dataset]
            all_entries.extend(hf_entries)
            print(f"✅ Added {len(hf_entries)} entries from Hugging Face")
        except Exception as e:
            print(f"⚠️  Could not load HF dataset: {e}")
        
        # 2. Load local datasets
        local_files = [
            "data/socratic_dialogues_dataset.json",
            "data/generated_dataset.json"
        ]
        
        for file_path in local_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    entries = data.get('dataset', [])
                    cleaned = [{"input": e.get("input", ""), "output": e.get("output", "")} for e in entries]
                    all_entries.extend(cleaned)
                    print(f"✅ Added {len(cleaned)} entries from {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️  Could not load {file_path}: {e}")
        
        # Save merged dataset
        merged_data = {"dataset": all_entries}
        os.makedirs("data", exist_ok=True)
        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Created merged dataset with {len(all_entries)} total entries")
    
    # Load the merged dataset
    with open(merged_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Dataset.from_list(data['dataset'])

# Create/load merged dataset
from datasets import Dataset
dataset = create_merged_dataset()
print(f"📊 Training with {len(dataset)} total entries from all sources")

# Function to clean speaker names from text
def clean_speaker_names(text):
    """Remove speaker names like 'SOCRATES:', 'MENO:', etc. from the beginning of text"""
    cleaned = re.sub(r'^[A-Z]{2,}:\s*', '', text.strip())
    return cleaned

# Use a standard instruction-following format for stability and better performance
def format_prompt(example):
    # Clean both input and output of speaker names
    clean_input = clean_speaker_names(example['input'])
    clean_output = clean_speaker_names(example['output'])
    text = f"<s>[INST] {clean_input} [/INST] {clean_output}</s>" # to let the model know the input and output/ format of LLaMa 2 and Mistral
    return {"text": text}

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=1024)

formatted_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names) # Removes the columns names and combines the input and output
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 5. Training ---
print("\nStep 5: Configuring and starting training...")
training_args = TrainingArguments(
    output_dir="./socrates-training-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit", # Memory-efficient optimizer, since we are using 4-bit quantization
    num_train_epochs=3,
    save_steps=200,
    logging_steps=25,
    learning_rate=5e-5, # Lowered learning rate for stability
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
    dataset_text_field="text",  # Add this line
    max_seq_length=1024,        # Add this line to fix the warning too
)
trainer.train()

# --- 6. Save ---
print(f"\nStep 6: Saving the fine-tuned LoRA adapters to {SFT_ADAPTER_OUTPUT_DIR}...")
trainer.model.save_pretrained(SFT_ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(SFT_ADAPTER_OUTPUT_DIR)

print("\n✅ Stage 1 Complete. The foundational Socrates model adapters are saved.")