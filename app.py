import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

import streamlit as st
import json
from config.prompts import code_template, summary_template
from core.model import query_ollama
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
    model_id = st.text_input("Ollama Model ID", value="deepseek-coder:6.7b-instruct")

    dataset_options = [
        "pglib_opf_case14_ieee",
        "pglib_opf_case118_ieee"
    ]
    selected_case = st.selectbox("Select OPF Dataset Case", dataset_options)

    if st.button("Load Model and Data"):
        try:
            dataset = OPFDataset(root='data', case_name=selected_case)
            st.session_state.data = dataset
            st.session_state.model_id = model_id
            st.session_state.model_loaded = True
            st.success(f"✅ Loaded dataset {selected_case}!")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# Main logic after loading
if st.session_state.model_loaded:
    st.subheader("💬 Ask a Question")
    query = st.text_area("Enter your prompt", height=150)

    if st.button("Run Query"):
        with st.spinner("⚙️ Running query through Ollama..."):
            summary, code, result_dict = run_pipeline(
                query=query,
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

        # Final summary
        st.success(f"✅ {summary}")
else:
    st.info("📂 Load a model and dataset to begin.")
