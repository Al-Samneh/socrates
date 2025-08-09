# 2_dpo_improvement.py - FINAL STABLE VERSION

print("Starting Stage 2: DPO Improvement Cycle...")
# --- 1. Imports ---
import torch
import re
import requests
import json
import os
from getpass import getpass
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer
from tqdm import tqdm

# --- 2. Securely Get API Key ---
try:
    GEMINI_API_KEY = getpass("What is your Gemini API key? ")
    if not GEMINI_API_KEY:
        raise ValueError("API Key cannot be empty.")
except (ValueError, Exception) as e:
    print(f"Could not get API key. Exiting. Error: {e}")
    exit()

# --- 3. Configuration ---
SFT_ADAPTER_DIR = "./socrates-sft-adapters"
DPO_ADAPTER_OUTPUT_DIR = "./socrates-dpo-adapters"
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# --- 4. Load Tokenizer with Robust Error Handling ---
print(f"\nStep 4a: Loading tokenizer...")

def load_tokenizer_safely():
    """Load tokenizer with multiple fallback strategies"""
    
    # Strategy 1: Try loading from base model directly (most reliable)
    try:
        print("Attempting to load tokenizer from base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        print("✅ Successfully loaded tokenizer from base model")
        return tokenizer
    except Exception as e:
        print(f"Failed to load from base model: {e}")
    
    # Strategy 2: Try with legacy flag
    try:
        print("Attempting to load with legacy settings...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False, legacy=True)
        print("✅ Successfully loaded tokenizer with legacy settings")
        return tokenizer
    except Exception as e:
        print(f"Failed with legacy settings: {e}")
    
    # Strategy 3: Force download and ignore cache
    try:
        print("Attempting fresh download...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, force_download=True, resume_download=False)
        print("✅ Successfully loaded tokenizer with fresh download")
        return tokenizer
    except Exception as e:
        print(f"Failed with fresh download: {e}")
        raise Exception("All tokenizer loading strategies failed. Please install sentencepiece: pip install sentencepiece")

tokenizer = load_tokenizer_safely()
tokenizer.pad_token = tokenizer.eos_token
# Left padding is required for generation
tokenizer.padding_side = "left"

# --- 4b. Load SFT Model and Configure for GENERATION ---
print(f"\nStep 4b: Loading the SFT model from {SFT_ADAPTER_DIR}...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

base_model_for_gen = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")

# Load the SFT adapters on top of the base model
model_to_improve = PeftModel.from_pretrained(base_model_for_gen, SFT_ADAPTER_DIR)

# *** FIX: REMOVED the merge_and_unload() call ***
# model_to_improve = model_to_improve.merge_and_unload() # This line was causing the error

# Set the model to evaluation mode for generation
model_to_improve.eval()
# --- 5. Generate Candidate Responses ---
print("\nStep 5: Generating candidate responses for judgment...")
eval_prompts = [
    "What is justice in the soul and in the state?",
    "Can a person truly know that they know nothing?",
    "What is the nature of virtue, and can it be taught?",
    "Is it ever right to commit an injustice for a greater good?",
    "What is the ideal form of government and who is fit to rule?",
    "Explain the allegory of the cave and its meaning.",
    "What is God?"
]
candidate_data = []

for prompt in tqdm(eval_prompts, desc="Generating Responses"):
    inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt", padding=True).to("cuda")
    
    # Generate for response 1
    generated_ids_1 = model_to_improve.generate(**inputs, max_new_tokens=350, temperature=0.75, do_sample=True)
    response_1 = tokenizer.decode(generated_ids_1[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    print(f"The first response: {response_1}")
    
    # Generate for response 2
    generated_ids_2 = model_to_improve.generate(**inputs, max_new_tokens=350, temperature=0.4, do_sample=True)
    response_2 = tokenizer.decode(generated_ids_2[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    print(f"The second response: {response_2}")

    candidate_data.append({"prompt": prompt, "response_1": response_1, "response_2": response_2})

# --- 6. Use Gemini API Judge to Create Preference Dataset ---
print("\nStep 6: Using Gemini 1.5 Flash via API to create preference data...")
del model_to_improve
torch.cuda.empty_cache()

preference_data = []
judge_prompt_template = """You are an expert in classical philosophy and the Socratic method. Your task is to evaluate two anonymous responses to a philosophical prompt. You must choose the response that better embodies the Socratic style.
Consider these criteria:
1.  **Questioning and Inquiry:** Does the response ask counter-questions or push the inquiry forward rather than just giving a final answer?
2.  **Intellectual Humility:** Does it acknowledge complexity or the limits of its own knowledge (e.g., "It seems to me," or "Let us examine this further")?
3.  **Logical consistency:** Is the reasoning clear and logical?
[PROMPT]: {prompt}
[RESPONSE 1]:
{response_1}
[RESPONSE 2]:
{response_2}
Based on the Socratic criteria, which response is superior? Your answer MUST be a single digit and nothing else: either '1' or '2'.
"""
headers = {'Content-Type': 'application/json', 'X-goog-api-key': GEMINI_API_KEY}

for item in tqdm(candidate_data, desc="Judging Responses with Gemini"):
    if not item['response_1'] or not item['response_2']: 
        continue
    prompt_for_judge = judge_prompt_template.format(**item)
    payload = {"contents": [{"parts": [{"text": prompt_for_judge}]}]}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        api_response = response.json()
        text_content = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        decision = re.search(r'\b(1|2)\b', text_content)
        if decision:
            chosen_idx = int(decision.group(0))
            print(f"The model chose: {chosen_idx}")
            if chosen_idx == 1:
                preference_data.append({"prompt": item['prompt'], "chosen": item['response_1'], "rejected": item['response_2']})
            else:
                preference_data.append({"prompt": item['prompt'], "chosen": item['response_2'], "rejected": item['response_1']})
    except Exception as e:
        print(f"\nAn error occurred during judgment: {e}")

if not preference_data:
    raise ValueError("Preference dataset is empty! Check API connection or Gemini's responses.")
preference_dataset = Dataset.from_list(preference_data)

# --- 7. DPO Training ---
print("\nStep 7: Starting DPO training with Gemini-judged preferences...")
# Flip padding side back to right for DPO training
tokenizer.padding_side = 'right'

dpo_base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")
lora_config = LoraConfig.from_pretrained(SFT_ADAPTER_DIR)
dpo_model = PeftModel.from_pretrained(dpo_base_model, SFT_ADAPTER_DIR, is_trainable=True, config=lora_config)

dpo_training_args = TrainingArguments(
    output_dir=DPO_ADAPTER_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    bf16=True,
    logging_steps=5,
    report_to="none"
)

# *** FIXED: Corrected the variable name typo ***
dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,
    args=dpo_training_args,  # Fixed: was dpo_training__args
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256
)

dpo_trainer.train()

# --- 8. Save Final DPO Adapters ---
print(f"\nStep 8: Saving the final, Gemini-judged DPO adapters to {DPO_ADAPTER_OUTPUT_DIR}...")
dpo_trainer.model.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)

print("\n✅ Stage 2 Complete. The improved Socrates DPO model adapters are saved.")