# app.py – Streamlit demo: RAG with Perplexity API
import os
import json
import io
import re
import time
import requests
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, BinaryIO

# ---------- Streamlit setup ----------
st.set_page_config(page_title="Perplexity RAG Demo", layout="wide")
st.title("🔍 RAG with Perplexity API")

# ---------- ENV / Keys ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY:
    st.error(
        "Add your Perplexity API key to a .env file as\n\n"
        "PERPLEXITY_API_KEY=pplx-...\n\nand restart the app."
    )
    st.stop()

# ---------- Model Configuration ----------
PERPLEXITY_MODELS = {
    "sonar": "Sonar",
    "sonar-pro": "Sonar Pro"
}

OPENAI_MODELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "gpt-3.5-turbo": "GPT-3.5 Turbo"
}

# API endpoints
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Supported file types
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "pdf", "txt"]

# ---------- Helpers ----------
def sanitize(name: str) -> str:
    """Return a safe table name."""
    return re.sub(r"\W+", "_", name.strip()) or "Sheet1"


def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Process an uploaded file and return its content as context."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    file_content = ""
    file_summary = ""
    
    try:
        if file_type in ["csv"]:
            df = pd.read_csv(uploaded_file)
            file_content = df.to_string()
            file_summary = f"CSV file with {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns)}"
            
        elif file_type in ["xlsx", "xls"]:
            xls = pd.ExcelFile(uploaded_file)
            sheets = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                sheets.append(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                file_content += f"\n\nSheet: {sheet_name}\n" + df.to_string()
            file_summary = f"Excel file with {len(sheets)} sheets: {'; '.join(sheets)}"
            
        elif file_type == "txt":
            file_content = uploaded_file.getvalue().decode("utf-8")
            file_summary = f"Text file with {len(file_content.splitlines())} lines"
            
        elif file_type == "pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                file_content = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
                file_summary = f"PDF file with {len(pdf_reader.pages)} pages"
            except ImportError:
                file_summary = "PDF file (PDF content extraction not available, install PyPDF2)"
                file_content = "PDF content extraction requires PyPDF2 library"
        else:
            file_content = "Unsupported file type"
            file_summary = f"Unsupported file type: {file_type}"
            
    except Exception as e:
        file_content = f"Error processing file: {str(e)}"
        file_summary = f"Error processing file: {str(e)}"
        
    return file_content, file_summary


def query_llm_api(provider: str, model_name: str, query: str, use_rag: bool = True, file_content: str = None, **kwargs) -> Tuple[Dict[str, Any], float]:
    """Query LLM API (Perplexity or OpenAI) with RAG support.
    
    Args:
        provider: API provider ('perplexity' or 'openai')
        model_name: Name of the model to use
        query: User query
        use_rag: Whether to use RAG (Sonar for Perplexity)
        file_content: Optional content from uploaded file
        **kwargs: Additional parameters for the API call
        
    Returns:
        Response from the API as a dictionary
    """
    # Prepare the message content
    if file_content:
        message_content = f"Document content:\n{file_content}\n\nUser query: {query}\n\nPlease answer the query based on the document content provided above."
    else:
        message_content = query
    
    # Common message structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": message_content}
    ]
    
    start_time = time.time()
    
    if provider == "perplexity":
        response = query_perplexity_api(model_name, messages, use_rag, **kwargs)
    elif provider == "openai":
        response = query_openai_api(model_name, messages, **kwargs)
    else:
        response = {"error": f"Unknown provider: {provider}"}
        
    elapsed_time = time.time() - start_time
    return response, elapsed_time


def query_perplexity_api(model_name: str, messages: List[Dict], use_rag: bool = True, **kwargs) -> Dict[str, Any]:
    """Query Perplexity API with RAG using Sonar Pro."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }
    
    # Set up the payload
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 40)
    }
    
    # Add Sonar Pro configuration if enabled
    if use_rag:
        payload["context"] = [
            {
                "type": "web_search",
                "enable": True,
                "include_citations": True,
                "include_raw_web_search": True
            }
        ]
    
    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Perplexity API request failed: {str(e)}")
        return {"error": str(e)}


def query_openai_api(model_name: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
    """Query OpenAI API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Set up the payload
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9)
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"OpenAI API request failed: {str(e)}")
        return {"error": str(e)}

def extract_response_text(response: Dict) -> str:
    """Extract the text response from the Perplexity API response."""
    try:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except (KeyError, IndexError):
        return "Error extracting response"

def extract_citations(response: Dict) -> List[Dict]:
    """Extract citations from the Perplexity API response."""
    try:
        # Get citations from the context output if available
        context_output = response.get("choices", [{}])[0].get("message", {}).get("context_output", {})
        web_search = context_output.get("web_search", {})
        citations = web_search.get("citations", [])
        
        # Format each citation
        formatted_citations = []
        for citation in citations:
            formatted_citations.append({
                "title": citation.get("title", "Unknown Source"),
                "url": citation.get("url", "#")
            })
        return formatted_citations
    except (KeyError, IndexError):
        return []

# ---------- UI ----------
st.sidebar.title("Model Settings")

# Model selection
st.sidebar.subheader("Select Models to Compare")

# Model 1 (Perplexity)
st.sidebar.markdown("#### Model 1: Perplexity")
model_1 = st.sidebar.selectbox(
    "Select Perplexity Model", 
    options=list(PERPLEXITY_MODELS.keys()),
    format_func=lambda x: PERPLEXITY_MODELS[x],
    index=1,  # Default to sonar-pro
    key="model_1"
)

# Model 2 (Provider selection)
st.sidebar.markdown("#### Model 2")
model_2_provider = st.sidebar.radio(
    "Select Provider",
    options=["perplexity", "openai"],
    format_func=lambda x: "Perplexity" if x == "perplexity" else "OpenAI",
    index=1,  # Default to OpenAI
    key="model_2_provider"
)

# Model 2 selection based on provider
if model_2_provider == "perplexity":
    model_2 = st.sidebar.selectbox(
        "Select Perplexity Model", 
        options=list(PERPLEXITY_MODELS.keys()),
        format_func=lambda x: PERPLEXITY_MODELS[x],
        index=0,  # Default to sonar
        key="model_2_perplexity"
    )
    model_2_display_name = PERPLEXITY_MODELS[model_2]
else:  # OpenAI
    model_2 = st.sidebar.selectbox(
        "Select OpenAI Model", 
        options=list(OPENAI_MODELS.keys()),
        format_func=lambda x: OPENAI_MODELS[x],
        index=0,  # Default to gpt-4o
        key="model_2_openai"
    )
    model_2_display_name = OPENAI_MODELS[model_2]

# RAG settings
st.sidebar.subheader("RAG Settings")
use_rag = st.sidebar.checkbox("Use RAG", value=True, help="For Perplexity, this enables Sonar Pro. For OpenAI, this will use the uploaded document if available.")

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    max_tokens = st.number_input("Max Tokens", min_value=16, max_value=4096, value=1024)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
    top_k = st.slider("Top K", min_value=1, max_value=100, value=40, step=1)

# File upload section
st.markdown("### Upload a document (Optional)")
uploaded_file = st.file_uploader(
    "Upload a file for document-based RAG", 
    type=SUPPORTED_FILE_TYPES,
    help="Supported file types: CSV, Excel, PDF, TXT"
)

file_content = None
file_summary = None

if uploaded_file:
    with st.spinner("Processing file..."):
        file_content, file_summary = process_uploaded_file(uploaded_file)
    st.success(f"File processed: {uploaded_file.name}")
    st.info(file_summary)

# Main content area
st.markdown("### Ask anything with RAG support")
query = st.text_area("Enter your query:", height=100)

if st.button("Submit") and query:
    # Set up columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # API parameters from user settings
    api_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    
    # Check if both API keys are available
    if not PERPLEXITY_API_KEY:
        st.error("Perplexity API key is missing. Please add it to your .env file.")
        st.stop()
        
    if model_2_provider == "openai" and not OPENAI_API_KEY:
        st.error("OpenAI API key is missing. Please add it to your .env file.")
        st.stop()
    
    with col1:
        st.markdown(f"#### {PERPLEXITY_MODELS[model_1]} Response")
        with st.spinner("Generating response..."):
            try:
                response1, response_time1 = query_llm_api(
                    provider="perplexity",
                    model_name=model_1,
                    query=query,
                    use_rag=use_rag and not file_content,  # Only use RAG if no file is uploaded
                    file_content=file_content,
                    **api_params
                )
                
                # Display the response text
                response_text = extract_response_text(response1)
                st.markdown(response_text)
                
                # Display response time
                st.info(f"Response time: {response_time1:.2f} seconds")
                
                # Display citations if available
                if use_rag:
                    citations = extract_citations(response1)
                    if citations:
                        st.markdown("##### Sources")
                        for i, citation in enumerate(citations):
                            st.markdown(f"{i+1}. [{citation['title']}]({citation['url']})")
            except Exception as e:
                st.error(f"Error with {model_1}: {str(e)}")
    
    with col2:
        st.markdown(f"#### {model_2_display_name} Response")
        with st.spinner("Generating response..."):
            try:
                response2, response_time2 = query_llm_api(
                    provider=model_2_provider,
                    model_name=model_2,
                    query=query,
                    use_rag=use_rag and not file_content,  # Only use RAG if no file is uploaded
                    file_content=file_content,
                    **api_params
                )
                
                # Display the response text
                response_text = extract_response_text(response2)
                st.markdown(response_text)
                
                # Display response time
                st.info(f"Response time: {response_time2:.2f} seconds")
                
                # Display citations if available
                if use_rag:
                    citations = extract_citations(response2)
                    if citations:
                        st.markdown("##### Sources")
                        for i, citation in enumerate(citations):
                            st.markdown(f"{i+1}. [{citation['title']}]({citation['url']})")
            except Exception as e:
                st.error(f"Error with {model_2}: {str(e)}")
                
    # Add response time comparison if both models responded successfully
    if 'response_time1' in locals() and 'response_time2' in locals():
        st.sidebar.markdown("### Response Time Comparison")
        time_diff = abs(response_time1 - response_time2)
        faster_model = PERPLEXITY_MODELS[model_1] if response_time1 < response_time2 else model_2_display_name
        st.sidebar.success(f"{faster_model} was faster by {time_diff:.2f} seconds")
else:
    st.info("👆 Enter a query and click 'Submit' to get responses from both models.")