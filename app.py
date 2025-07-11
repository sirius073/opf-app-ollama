import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

import streamlit as st
import json
from config.prompts import code_template, summary_template
from core.model import query_ollama, refine_query_with_llm, load_phi2_electrical_model
from core.executor import run_pipeline
from torch_geometric.datasets import OPFDataset

st.set_page_config(page_title="Power Grid LLM Interface", layout="wide")
st.title("🔌 Power Grid Code Assistant with Ollama")

# Session setup
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Sidebar for model + data loading
with st.sidebar:
    st.header("Configuration")
    model_id = st.text_input("Ollama Model ID", value="deepseek-coder:33b-instruct")

    dataset_options = [
        "pglib_opf_case14_ieee",
        "pglib_opf_case118_ieee"
    ]
    selected_case = st.selectbox("Select OPF Dataset Case", dataset_options)

    if st.button("Load Model and Data"):
        try:
            # Load OPF dataset
            dataset = OPFDataset(root='data', case_name=selected_case)
            st.session_state.data = dataset
            st.session_state.model_id = model_id

            # # Load Phi-2 fine-tuned model only once
            # with st.spinner("🔌 Loading Phi-2 Electrical Model..."):
            #     phi_model, phi_tokenizer = load_phi2_electrical_model()
            #     st.session_state.phi_model = phi_model
            #     st.session_state.phi_tokenizer = phi_tokenizer

            st.session_state.model_loaded = True
            st.success(f"✅ Loaded dataset {selected_case} and models!")
        except Exception as e:
            st.error(f"❌ Error loading dataset or models: {e}")

# Main logic after loading
if st.session_state.model_loaded:
    st.subheader("💬 Ask a Question")
    query = st.text_area("Enter your prompt", height=150)
    use_refinement = st.checkbox("🔍 Refine query using Phi-2 Electrical Engineering model")

    if st.button("Run Query"):
        final_query = query
        refined_instruction = None

        # Apply query refinement if selected
        if use_refinement:
            with st.spinner("🧠 Refining query using Phi-2..."):
                try:
                    refined_instruction = refine_query_with_llm(
                        query, 
                        st.session_state.phi_model, 
                        st.session_state.phi_tokenizer
                    )
                    # Combine user query + instruction for final query to Ollama
                    final_query = f"{query}\n\nInstruction: {refined_instruction}"
                    st.subheader("🧾 Refined Instruction (Phi-2):")
                    st.code(refined_instruction)
                except Exception as e:
                    st.warning(f"⚠️ Refinement failed, using original query.\n{e}")
                    final_query = query
                   
        with st.spinner("⚙️ Running query through Ollama..."):
            summary, code, result_dict = run_pipeline(
                query=final_query,
                dataset=st.session_state.data,
                model_id=st.session_state.model_id
            )

        st.subheader("🧠 Generated Code")
        st.code(code, language="python")

        st.subheader("📦 Result Dictionary")
        st.json(result_dict)

        # 🔹 Handle multiple plots
        if "plots" in result_dict:
            all_plots = result_dict["plots"]
            if all_plots:
                st.subheader("📊 Plots")
                if not isinstance(all_plots, list):
                    all_plots = [all_plots]
                for i, fig in enumerate(all_plots):
                    st.markdown(f"**Plot {i+1}**")
                    st.pyplot(fig)

        # 🔸 Handle single plot
        elif "plot" in result_dict:
            try:
                st.pyplot(result_dict["plot"])
            except:
                st.plotly_chart(result_dict["plot"])

        st.success(f"✅ {summary}")
else:
    st.info("📂 Load a model and dataset to begin.")
