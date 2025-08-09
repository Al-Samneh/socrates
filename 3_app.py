# 3_app.py
# Stage 3: Interactive Chat Application

print("Starting Stage 3: Loading model for deployment...")

# --- 1. Imports ---
import torch
import threading
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 2. Configuration & Model Loading ---
DPO_ADAPTER_DIR = "./socrates-dpo-adapters" # Load our best, DPO-tuned adapters
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Use 4-bit quantization for efficient inference
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto")
inference_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR)
inference_model = inference_model.merge_and_unload()
inference_model.eval()
tokenizer = AutoTokenizer.from_pretrained(DPO_ADAPTER_DIR)

print("✅ Model loaded and ready for chat.")

# --- 3. Backend API (FastAPI) ---
api = FastAPI()
class PromptRequest(BaseModel):
    prompt: str

@api.post("/generate")
async def generate_response(request: PromptRequest):
    input_text = f"<s>[INST] {request.prompt} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = inference_model.generate(**inputs, max_new_tokens=250, temperature=0.7, do_sample=True)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    return {"response": response_text}

# --- 4. Frontend UI (Streamlit) ---
def run_streamlit():
    st.set_page_config(page_title="Chat with Socrates", page_icon="🏛️")
    st.title("🏛️ Chat with Socrates")
    st.caption("An LLM iteratively improved with an LLM Judge and DPO.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Greetings. What shall we inquire into today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Pose your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("The master is thinking..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt}, timeout=120)
                    response.raise_for_status()
                    socrates_response = response.json()["response"]
                except requests.exceptions.RequestException as e:
                    socrates_response = f"Alas, my thoughts are clouded. Error: {e}"
            st.markdown(socrates_response)
            st.session_state.messages.append({"role": "assistant", "content": socrates_response})

# --- 5. Main execution block to run both apps ---
if __name__ == "__main__":
    # Run FastAPI in a separate thread
    def run_api():
        uvicorn.run(api, host="0.0.0.0", port=8000)
    
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Run Streamlit
    run_streamlit()