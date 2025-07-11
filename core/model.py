import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def query_ollama(prompt, model="deepseek-coder:33b-instruct"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()  
    except requests.exceptions.RequestException as e:
        print(f"[Ollama Error] {e}")
        return f"ERROR: {str(e)}"

def load_phi2_electrical_model():
    base_model = "microsoft/phi-2"
    adapter_model = "STEM-AI-mtl/phi-2-electrical-engineering"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_model)
    model.eval()
    return model, tokenizer

def refine_query_with_llm(user_query, model, tokenizer):
    prompt = f"""
### Instruction:
You are an Electrical and Power System expert.
You are given a user query for a particular `dataset` that is a power system simulation with each `data` object in the dataset representing a different loading system, 
Convert it into a precise technical instruction with steps. You don't have to give any code just the steps.
NOTE:-
- The dataset is already loaded.
- Do not mention accessing the dataset.
- Do not instruct to print anything and instead just store it in 'result' dictionary.
User Query: {user_query}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=1)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean the response
    response_part = decoded_output.split("### Response:")[-1].split("```")[0].strip()
    return response_part

