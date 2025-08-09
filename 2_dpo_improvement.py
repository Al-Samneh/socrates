# 2_dpo_improvement_gemini.py
# Stage 2: Improvement with Gemini 1.5 Pro as Judge and DPO

print("Starting Stage 2: DPO Improvement Cycle with Gemini Judge...")
# --- 1. Imports ---
import torch
import re
import requests # To make API calls
import json
from getpass import getpass # To securely ask for your API key
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer
from tqdm import tqdm

# --- 2. Securely Get API Key ---
try:
    # Ask for the API key in a hidden prompt
    GEMINI_API_KEY = 'AIzaSyAszi5Ing9J5rKPg8VpKkjLN2iN7P2KTv0'
    if not GEMINI_API_KEY:
        raise ValueError("API Key cannot be empty.")
except (ValueError, Exception) as e:
    print(f"Could not get API key. Exiting. Error: {e}")
    exit()

# --- 3. Configuration ---
SFT_ADAPTER_DIR = "./socrates-sft-adapters"
DPO_ADAPTER_OUTPUT_DIR = "./socrates-dpo-adapters"
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# We'll use the powerful Gemini 2.0 Flash for judging
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- 4. Load SFT Model to Generate Responses ---
print(f"\nStep 4: Loading the SFT model from {SFT_ADAPTER_DIR}...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

base_model_for_gen = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")
model_to_improve = PeftModel.from_pretrained(base_model_for_gen, SFT_ADAPTER_DIR)
model_to_improve = model_to_improve.merge_and_unload()
model_to_improve.eval()
tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR)

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
    input_text = f"<s>[INST] {prompt} [/INST]"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    response_1 = tokenizer.decode(model_to_improve.generate(**input_ids, max_new_tokens=250, temperature=0.75, do_sample=True)[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    response_2 = tokenizer.decode(model_to_improve.generate(**input_ids, max_new_tokens=250, temperature=0.4, do_sample=True)[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    candidate_data.append({"prompt": prompt, "response_1": response_1, "response_2": response_2})

# --- 6. Use Gemini API Judge to Create Preference Dataset ---
print("\nStep 6: Using Gemini 2.0 Flash via API to create preference data...")
del model_to_improve # Free up GPU memory since the judge is external
torch.cuda.empty_cache()

preference_data = []
# This is a highly specific prompt engineered to get a clean "1" or "2" from the API
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
    prompt_for_judge = judge_prompt_template.format(**item)
    payload = {"contents": [{"parts": [{"text": prompt_for_judge}]}]}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Safely extract text from the Gemini API response
        api_response = response.json()
        text_content = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        
        decision = re.search(r'\b(1|2)\b', text_content)
        if decision:
            chosen_idx = int(decision.group(0))
            if chosen_idx == 1:
                preference_data.append({"prompt": item['prompt'], "chosen": item['response_1'], "rejected": item['response_2']})
            else:
                preference_data.append({"prompt": item['prompt'], "chosen": item['response_2'], "rejected": item['response_1']})
    except requests.exceptions.RequestException as e:
        print(f"\nAPI request failed: {e}")
    except (KeyError, IndexError) as e:
        print(f"\nCould not parse Gemini response: {api_response}. Error: {e}")

if not preference_data:
    raise ValueError("Preference dataset is empty! Check API connection or Gemini's responses.")
preference_dataset = Dataset.from_list(preference_data)

# --- 7. DPO Training ---
print("\nStep 7: Starting DPO training with Gemini-judged preferences...")

# --- Part 1: Configure the tokenizer (This is ESSENTIAL) ---
# The DPOTrainer will implicitly use this tokenizer object later.
# It needs a padding token to create batches of the same length.
tokenizer.pad_token = tokenizer.eos_token

# Optional but good practice for some models to avoid issues with attention
# tokenizer.padding_side = 'left' 

# --- Part 2: Load the models (No changes here) ---
dpo_base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, 
    quantization_config=bnb_config, 
    device_map="auto"
)
lora_config = LoraConfig.from_pretrained(SFT_ADAPTER_DIR)
dpo_model = PeftModel.from_pretrained(
    dpo_base_model, 
    SFT_ADAPTER_DIR, 
    is_trainable=True, 
    config=lora_config
)

# --- Part 3: Define Training Arguments (No changes here) ---
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

# --- Part 4: Initialize the DPOTrainer (CORRECTED for your version) ---
# Pass the required length arguments, but DO NOT pass the tokenizer object.
dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,
    args=dpo_training_args,
    train_dataset=preference_dataset,
    # These length arguments are critical for the internal data collator
    # max_length=512,
    # max_prompt_length=256 
)

dpo_trainer.train()


# --- 8. Save Final DPO Adapters ---
print(f"\nStep 8: Saving the final, Gemini-judged DPO adapters to {DPO_ADAPTER_OUTPUT_DIR}...")
dpo_trainer.model.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)

print("\n✅ Stage 2 Complete. The improved Socrates DPO model adapters are saved.")