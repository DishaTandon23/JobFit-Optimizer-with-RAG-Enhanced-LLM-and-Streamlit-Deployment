import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import subprocess
import os

# Set page configuration
st.set_page_config(
    page_title="JobFit Optimizer",
    page_icon="💼",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTitle { color: #2E4057; font-size: 2.5rem !important; }
    .skill-item { 
        padding: 0.5rem; margin: 0.3rem 0; border-radius: 5px; 
        background-color: #f0f2f6; color: rgb(49, 51, 63); 
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Load CSV data
@st.cache_data
def load_csv():
    return pd.read_csv("postings.csv")  # Replace with the path to your job postings CSV

# Create vector store
@st.cache_resource
def create_vector_store(data):
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    texts = data['description'].fillna('').tolist()
    embeddings = encoder.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, texts, encoder

def search_similar_texts(query, index, texts, encoder, k=3):
    """Retrieve the top K similar job descriptions."""
    query_vector = encoder.encode([query])
    D, I = index.search(query_vector.astype('float32'), k=k)
    return [texts[i] for i in I[0]]

def get_skills_from_llm(job_title, similar_texts):
    """Generate skills dynamically using LLM based on retrieved job descriptions."""
    # Combine similar texts as context
    context = "\n\n".join(similar_texts)
    full_prompt = (
        f"You are an AI expert at extracting skills for job roles. Below are job descriptions for similar roles:\n\n"
        f"{context}\n\n"
        f"Based on the above, list the top 5 most relevant skills required for the job title: '{job_title}'."
        f"Provide only the skills in a numbered list."
    )

    # Call LLM using subprocess (e.g., Ollama CLI)
    command = ["ollama", "run", "llama3.1:8b"]
    result = subprocess.run(command, input=full_prompt, text=True, stdout=subprocess.PIPE)
    return result.stdout.strip()

def main():
    st.title("JobFit Optimizer: Skill Predictor")
    st.write("Enter a job title to predict the top 5 required skills dynamically.")
    
    # Load data and create vector store at startup
    if st.session_state.vector_store is None:
        with st.spinner("Initializing... Please wait."):
            data = load_csv()
            index, texts, encoder = create_vector_store(data)
            st.session_state.vector_store = (index, texts, encoder)
    
    # Input field for job title
    job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    
    if job_title:
        index, texts, encoder = st.session_state.vector_store

        # Step 1: Retrieve similar job descriptions
        with st.spinner("Retrieving relevant job descriptions..."):
            similar_texts = search_similar_texts(job_title, index, texts, encoder)
        
        # Step 2: Generate skills using the LLM
        with st.spinner("Extracting skills using LLM..."):
            skills_output = get_skills_from_llm(job_title, similar_texts)

        # Step 3: Display results
        st.subheader("Top 5 Recommended Skills:")
        if skills_output:
            skills_list = skills_output.split("\n")  # Parse LLM output into a list
            for skill in skills_list:
                st.markdown(f"""
                    <div class="skill-item">
                        <span>{skill.strip()}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("The LLM did not return any skills. Please try again.")

if __name__ == "__main__":
    main()
