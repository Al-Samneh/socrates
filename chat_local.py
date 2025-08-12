#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel    
import time
import os

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print a beautiful SAMNEH AI banner"""
    banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████   █████   ███    ███ ███    ██ ███████ ██   ██      █████  ██       ║
║   ██       ██   ██  ████  ████ ██ ██   ██ ██      ██   ██     ██   ██ ██       ║
║   ███████  ███████  ██ ████ ██ ██  ██  ██ █████   ███████     ███████ ██       ║
║        ██  ██   ██  ██  ██  ██ ██   ██ ██ ██      ██   ██     ██   ██ ██       ║
║   ███████  ██   ██  ██      ██ ██    ████ ███████ ██   ██     ██   ██ ██       ║
║                                                                               ║
║                     🧠 SAMNEH AI - PHILOSOPHICAL COMPANION 🧠                 ║
║                          Powered by Advanced ML Training                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print("\033[36m" + banner + "\033[0m")  # Cyan color

def print_loading_animation(text):
    """Print loading animation"""
    print(f"\033[33m⚡ {text}\033[0m", end="")
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(" ✅")

def load_socrates_model():
    """Load the fine-tuned Socrates model"""
    base_model = "mistralai/Mistral-7B-v0.1"
    
    print_loading_animation("Initializing Socrates AI")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    print_loading_animation("Loading neural pathways")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print_loading_animation("Integrating philosophical wisdom")
    model = PeftModel.from_pretrained(model, "./socrates-dpo-adapters")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("\033[32m🚀 GPU acceleration enabled\033[0m")
    else:
        print("\033[33m💻 Running on CPU\033[0m")
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("\033[32m✨ Socrates AI is ready for philosophical dialogue!\033[0m\n")
    return model, tokenizer

def print_separator():
    """Print a decorative separator"""
    print("\033[34m" + "─" * 80 + "\033[0m")

def chat_with_socrates(model, tokenizer):
    """Interactive chat loop"""
    
    # Welcome message
    welcome_box = """
╔════════════════════════════════════════════════════════════════════════════╗
║  🎓 Welcome to your personal Socratic dialogue experience!                ║
║                                                                            ║
║  💭 Ask deep questions about life, virtue, knowledge, and existence       ║
║  🤔 Challenge assumptions and explore philosophical concepts               ║
║  📚 Engage in the ancient art of dialectical reasoning                    ║
║                                                                            ║
║  Type 'quit', 'exit', or 'bye' to end the conversation                    ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    print("\033[35m" + welcome_box + "\033[0m")
    
    conversation_count = 0
    
    while True:
        conversation_count += 1
        
        # User input with nice prompt
        print(f"\033[36m┌─ Question #{conversation_count}\033[0m")
        question = input("\033[36m│ 🤷 You: \033[0m")
        
        if question.lower() in ['quit', 'exit', 'bye']:
            farewell_box = """
╔════════════════════════════════════════════════════════════════════════════╗
║  🎓 "The unexamined life is not worth living." - Socrates                 ║
║                                                                            ║
║  Thank you for this philosophical journey. May wisdom guide your path!    ║
║                                                                            ║
║  🌟 SAMNEH AI - Where Ancient Wisdom Meets Modern Technology 🌟          ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
            print("\033[35m" + farewell_box + "\033[0m")
            break
        
        print("\033[36m└─\033[0m")
        
        # Thinking animation
        print("\033[33m🧠 Socrates is contemplating", end="")
        for i in range(4):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print("\033[0m")
        
        # Format as instruction
        prompt = f"<s>[INST] {question} [/INST]"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        # Format response beautifully
        print("\033[32m┌─ Socratic Response\033[0m")
        
        # Word wrap the response
        words = response.split()
        lines = []
        current_line = ""
        max_width = 70
        
        for word in words:
            if len(current_line + " " + word) <= max_width:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Print wrapped response
        for i, line in enumerate(lines):
            if i == 0:
                print(f"\033[32m│ 🎓 Socrates: \033[0m{line}")
            else:
                print(f"\033[32m│            \033[0m{line}")
        
        print("\033[32m└─\033[0m")
        print()  # Empty line for spacing

def main():
    """Main function"""
    clear_screen()
    print_banner()
    
    try:
        model, tokenizer = load_socrates_model()
        chat_with_socrates(model, tokenizer)
    except KeyboardInterrupt:
        print("\n\n\033[33m⚠️  Conversation interrupted by user\033[0m")
        print("\033[35m🌟 Thanks for using SAMNEH AI! 🌟\033[0m")
    except Exception as e:
        print(f"\n\033[31m❌ Error: {e}\033[0m")
        print("\033[33m💡 Please check your setup and try again\033[0m")

if __name__ == "__main__":
    main()