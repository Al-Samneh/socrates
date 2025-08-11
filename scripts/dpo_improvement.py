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

def generate_evaluation_prompts(num_prompts=500):
    """Generate a diverse set of philosophical prompts for evaluation"""
    
    # Base philosophical topics and question patterns
    philosophical_topics = [
        "justice", "virtue", "knowledge", "truth", "beauty", "good", "evil", "wisdom", 
        "courage", "temperance", "piety", "friendship", "love", "death", "soul", 
        "reality", "appearance", "being", "becoming", "forms", "ideas", "mind", 
        "body", "free will", "determinism", "morality", "ethics", "politics", 
        "government", "law", "education", "art", "science", "mathematics", 
        "language", "consciousness", "identity", "time", "space", "causation",
        "existence", "essence", "meaning", "purpose", "happiness", "pleasure",
        "pain", "honor", "shame", "pride", "humility", "faith", "reason"
    ]
    
    question_patterns = [
        "What is the nature of {topic}?",
        "Can {topic} be taught or learned?", 
        "How do we distinguish true {topic} from false {topic}?",
        "What is the relationship between {topic} and virtue?",
        "Is {topic} the same for all people or does it vary?",
        "What role does {topic} play in the good life?",
        "How does {topic} relate to knowledge and ignorance?",
        "Can we have certainty about {topic}?",
        "What are the different types or forms of {topic}?",
        "How do we acquire understanding of {topic}?",
        "What is the highest form of {topic}?",
        "Is {topic} innate or learned through experience?",
        "How does {topic} manifest in the soul versus the state?",
        "What are the dangers of misunderstanding {topic}?",
        "How do we examine our beliefs about {topic}?"
    ]
    
    # Generate base prompts from combinations
    prompts = []
    import random
    
    # Generate systematic combinations
    for topic in philosophical_topics:
        for pattern in question_patterns[:3]:  # Use first 3 patterns for each topic
            prompts.append(pattern.format(topic=topic))
    
    # Add classical Socratic questions
    classical_prompts = [
        "What is justice in the soul and in the state?",
        "Can a person truly know that they know nothing?", 
        "What is the nature of virtue, and can it be taught?",
        "Is it ever right to commit an injustice for a greater good?",
        "What is the ideal form of government and who is fit to rule?",
        "Explain the allegory of the cave and its meaning.",
        "What distinguishes the philosopher from other types of people?",
        "How do we know if our beliefs are true or merely opinions?",
        "What is the relationship between knowledge and virtue?",
        "Can we ever be certain about moral truths?",
        "What makes a life worth living?",
        "How should we prepare for death?",
        "What is the role of emotion in moral decision-making?",
        "Are there absolute moral truths or is morality relative?",
        "What is the proper relationship between the individual and society?",
        "How do we distinguish between appearance and reality?",
        "What is the source of moral authority?",
        "Can wisdom be separated from virtue?",
        "What is the nature of the soul?",
        "How do we overcome ignorance and achieve knowledge?"
    ]
    
    prompts.extend(classical_prompts)
    
    # Add complex comparative questions
    comparative_prompts = [
        "Is it better to suffer injustice or to commit it?",
        "Which is more valuable: knowledge or virtue?",
        "Should we trust reason or experience more?",
        "Is the examined life always worth living?",
        "Are philosophers fit to rule, or should they avoid politics?",
        "Is it possible to do wrong knowingly?",
        "Which matters more: intention or outcome in moral acts?",
        "Should we fear death or embrace it as natural?",
        "Is happiness a feeling or a way of living?",
        "Are the gods moral, or is morality independent of the gods?"
    ]
    
    prompts.extend(comparative_prompts)
    
    # Shuffle and select the requested number
    random.shuffle(prompts)
    return prompts[:num_prompts]




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

# --- 5. Generate 500 Candidate Responses for Better Accuracy ---
print("\nStep 5: Generating 500 candidate response pairs for comprehensive evaluation...")

# Generate 500 diverse philosophical prompts instead of just 7
eval_prompts = generate_evaluation_prompts(500)
print(f"Generated {len(eval_prompts)} evaluation prompts for comprehensive testing")

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

# Updated judge prompt template with rating scale (keeping binary choice commented for reference)
judge_prompt_template = """You are an expert in classical philosophy and the Socratic method. Your task is to evaluate two anonymous responses to a philosophical prompt and rate their quality on the Socratic criteria.

Evaluate each response on these criteria (1-5 scale for each):

1. **Questioning and Inquiry (1-5):** Does the response ask counter-questions or push the inquiry forward rather than just giving final answers? (1=gives direct answers only, 5=masterfully uses questions to guide thinking)

2. **Intellectual Humility (1-5):** Does it acknowledge complexity or the limits of its own knowledge? Uses phrases like "It seems to me," "Let us examine," "I may be wrong." (1=claims certainty on complex topics, 5=demonstrates appropriate epistemic humility)

3. **Logical Consistency (1-5):** Is the reasoning clear, coherent, and free from contradictions? (1=contradictory or unclear, 5=perfectly logical and clear)

4. **Depth of Analysis (1-5):** Does it break down complex concepts and examine underlying assumptions? (1=superficial treatment, 5=profound philosophical analysis)

5. **Engagement Style (1-5):** Does it invite further dialogue and thinking rather than closing off discussion? (1=conversation-ending, 5=opens new avenues of inquiry)

[PROMPT]: {prompt}

[RESPONSE 1]:
{response_1}

[RESPONSE 2]: 
{response_2}

Rate each response on the 5 criteria above. Then provide an overall assessment.

Format your response exactly as follows:
RESPONSE 1 SCORES: [Q&I: X/5, Humility: X/5, Logic: X/5, Depth: X/5, Engagement: X/5]
RESPONSE 2 SCORES: [Q&I: X/5, Humility: X/5, Logic: X/5, Depth: X/5, Engagement: X/5]
BETTER RESPONSE: [1 or 2]

Note: We use granular ratings (1-5 scale) instead of simple binary choice because it provides richer feedback for training the model to understand the nuances of Socratic dialogue. The detailed scores help the DPO algorithm learn more precisely which aspects of responses are better or worse.
"""

# COMMENTED OUT: Original binary choice template
# This was the original simpler approach, but we upgraded to rating scale for better granularity:
"""
# Original binary judge template (keeping for reference):
# judge_prompt_template = '''You are an expert in classical philosophy and the Socratic method. 
# Your task is to evaluate two anonymous responses to a philosophical prompt. You must choose 
# the response that better embodies the Socratic style.
# Consider these criteria:
# 1. **Questioning and Inquiry:** Does the response ask counter-questions or push the inquiry forward?
# 2. **Intellectual Humility:** Does it acknowledge complexity or limits of knowledge?
# 3. **Logical consistency:** Is the reasoning clear and logical?
# Based on the Socratic criteria, which response is superior? Your answer MUST be a single digit: either '1' or '2'.
# '''
# 
# Why we moved away from binary choice:
# - Binary choice loses important nuance about WHY one response is better
# - Rating scales provide richer training signal for DPO algorithm  
# - Detailed scores help model learn specific aspects of good Socratic dialogue
# - More granular feedback leads to better model improvements
"""
headers = {'Content-Type': 'application/json', 'X-goog-api-key': GEMINI_API_KEY}

def extract_ratings_and_choice(gemini_response):
    """Extract ratings and final choice from Gemini's detailed response"""
    import re
    
    try:
        # Extract Response 1 scores
        r1_match = re.search(r'RESPONSE 1 SCORES:.*?\[(.*?)\]', gemini_response, re.DOTALL)
        # Extract Response 2 scores  
        r2_match = re.search(r'RESPONSE 2 SCORES:.*?\[(.*?)\]', gemini_response, re.DOTALL)
        # Extract final choice
        choice_match = re.search(r'BETTER RESPONSE:\s*(\d)', gemini_response)
        
        if r1_match and r2_match and choice_match:
            r1_scores = r1_match.group(1)
            r2_scores = r2_match.group(1) 
            choice = int(choice_match.group(1))
            
            # Calculate average scores for additional data
            def extract_avg_score(score_text):
                numbers = re.findall(r'(\d+)/5', score_text)
                if numbers:
                    return sum(int(n) for n in numbers) / len(numbers)
                return 0
            
            r1_avg = extract_avg_score(r1_scores)
            r2_avg = extract_avg_score(r2_scores)
            
            return {
                'choice': choice,
                'response_1_scores': r1_scores,
                'response_2_scores': r2_scores, 
                'response_1_avg': r1_avg,
                'response_2_avg': r2_avg,
                'score_difference': abs(r1_avg - r2_avg)
            }
    except Exception as e:
        print(f"Error parsing ratings: {e}")
    
    # Fallback to simple choice extraction if detailed parsing fails
    simple_choice = re.search(r'\b(1|2)\b', gemini_response)
    if simple_choice:
        return {'choice': int(simple_choice.group(0)), 'fallback': True}
    
    return None

# Update the main evaluation loop:
for item in tqdm(candidate_data, desc="Judging Responses with Gemini (Detailed Ratings)"):
    if not item['response_1'] or not item['response_2']: 
        continue
        
    prompt_for_judge = judge_prompt_template.format(**item)
    payload = {"contents": [{"parts": [{"text": prompt_for_judge}]}]}
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        api_response = response.json()
        text_content = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        
        # Extract detailed ratings and choice
        judgment = extract_ratings_and_choice(text_content)
        
        if judgment and 'choice' in judgment:
            chosen_idx = judgment['choice']
            print(f"Gemini chose response {chosen_idx} (Score diff: {judgment.get('score_difference', 'N/A'):.2f})")
            
            if chosen_idx == 1:
                preference_item = {
                    "prompt": item['prompt'], 
                    "chosen": item['response_1'], 
                    "rejected": item['response_2'],
                    "judgment_details": judgment  # Store detailed ratings for analysis
                }
            else:
                preference_item = {
                    "prompt": item['prompt'], 
                    "chosen": item['response_2'], 
                    "rejected": item['response_1'],
                    "judgment_details": judgment
                }
            
            preference_data.append(preference_item)
            
    except Exception as e:
        print(f"\nError during judgment: {e}")

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



# --- 9. Quick Model Testing ---
print("\nStep 9: Testing the trained model...")

def test_socratic_model():
    """Quick test of the trained model"""
    test_prompts = [
        "What is justice?",
        "Can virtue be taught?", 
        "How do we know what we know?",
        "What is God?",
        "What is the relationship between law, custom, and virtue?",
        "Is education essential?",
        "What will superintelligence be like?"
    ]
    
    # Load the final trained model
    tokenizer.padding_side = "left"  # For generation
    
    print("\n🧠 Testing trained Socratic model:")
    print("="*50)
    
    for prompt in test_prompts:
        inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = dpo_trainer.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        print(f"\n💭 Question: {prompt}")
        print(f"🎯 Socrates: {response}")
        print("-"*30)

# Run the test
test_socratic_model()