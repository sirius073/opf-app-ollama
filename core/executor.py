import re
import streamlit as st
import json
import torch
from torch_geometric.data import HeteroData
from core.model import query_ollama
from config.prompts import code_template, summary_template, fix_prompt as fix_prompt_template

# ✅ Utility: extract <code>...</code> or ```...``` block
def extract_code_block(text: str) -> str:
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL) or \
            re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# ✅ Utility: convert PyTorch objects to JSON-safe format
def make_serializable(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, dict):
        return {k: make_serializable(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [make_serializable(vv) for vv in v]
    return v

# ✅ Main pipeline
def run_pipeline(query: str, dataset: HeteroData, model_id: str):
    result = {}
    torch.cuda.empty_cache()
    max_attempts = 3

    # ✅ Step 1: Get code from LLM
    llm_code_output = query_ollama(code_template.format(query=query), model_id)
    code_block = extract_code_block(llm_code_output)

    if not code_block:
        return "Code not found", "", {}

    attempt = 0
    error_message = ""
    while attempt < max_attempts:
        try:
            exec_scope = {
                "dataset": dataset,
                "result": result,
                "torch": torch,
                "st": st,
            }
            exec(code_block, exec_scope)
            result = exec_scope.get("result", {})
            break
        except Exception as e:
            error_message = str(e)
            attempt += 1
            if attempt >= max_attempts:
                return f"Execution error after {max_attempts} attempts: {error_message}", code_block, {}

            # ✅ Step 2: Fix code via LLM
            st.warning(f"❌ Attempt {attempt} failed: {error_message}")
            st.info("🛠️ LLM is attempting to fix the code...")
            retry_prompt = fix_prompt_template.format(error_message=error_message, code_block=code_block)
            fixed_output = query_ollama(retry_prompt, model_id)
            code_block = extract_code_block(fixed_output)

    # ✅ Step 3: Convert result to serializable
    serializable_result = {
        k: make_serializable(v)
        for k, v in result.items()
        if k not in ["plot", "plots"]
    }

    # ✅ Step 4: Ask for summary
    summary_raw = query_ollama(
        summary_template.format(
            query=query,
            result=json.dumps(serializable_result, indent=2)
        ),
        model_id
    )
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_raw, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary not found."

    return summary, code_block, result
