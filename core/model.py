import requests

def query_ollama(prompt, model="deepseek-coder:33b-instruct"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60 
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()  
    except requests.exceptions.RequestException as e:
        print(f"[Ollama Error] {e}")
        return f"ERROR: {str(e)}"
