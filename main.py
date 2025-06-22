import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client
client = InferenceClient(token=HF_TOKEN)

# Available models
MODELS = {
    "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama": "meta-llama/Llama-3.1-8B-Instruct"
}

# UI Config
st.set_page_config(
    page_title="AlgoForge - AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        line-height: 1.6;
    }
    .user {
        background-color: #020e24;
        border-left: 5px solid #4285f4;
    }
    .bot {
        background-color: #04495c;
        border-left: 5px solid #34a853;
    }
    code {
        background-color: #eeeeee;
        padding: 2px 4px;
        border-radius: 4px;
        font-size: 90%;
    }
    pre {
        background-color: #eeeeee;
        padding: 1rem;
        border-radius: 10px;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AlgoForge - AI Assistant")

# Sidebar
with st.sidebar:
    st.header("🧠 AlgoForge Controls")

    selected_model_name = st.selectbox(
        "Select a model:",
        list(MODELS.keys()),
        index=0
    )
    selected_model = MODELS[selected_model_name]

    st.markdown("---")

    st.subheader("Chat History")
    clear_chat = st.button("🗑️ Clear Chat")

# Initialize session state
if "messages" not in st.session_state or clear_chat:
    st.session_state.messages = []

# Function to format messages
def format_message(sender, message, is_user=False):
    css_class = "user" if is_user else "bot"
    icon = "🧑‍💻" if is_user else "🤖"
    name = "You" if is_user else "AlgoForge"
    return f"""
    <div class='message {css_class}'>
        <strong>{icon} {name}</strong><br>
        {message}
    </div>
    """

# Generate response
def generate_response(prompt):
    template = f"<|user|>\n{prompt}\n<|assistant|>"
    response = client.text_generation(
        prompt=template,
        model=selected_model,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        stop_sequences=["<|user|>"]
    )
    return response.strip()

# Main Chat UI
with st.container():
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        st.markdown(format_message(msg["role"], msg["content"], is_user), unsafe_allow_html=True)
    

    prompt = st.chat_input("Ask AlgoForge anything...")
    # Example Prompts Section
with st.expander("💡 Example Prompts"):
    st.markdown("""
    - Why is my DFS not visiting all nodes?  
    - Can you help me optimize my Binary Search?  
    - What’s the difference between BFS and Dijkstra?  
    - Explain Kadane’s Algorithm with code.  
    - What is the time complexity of Merge Sort?  
    - Write Python code to detect a cycle in a graph.  
    """)


    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(format_message("user", prompt, is_user=True), unsafe_allow_html=True)

        # Generate assistant reply
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(format_message("assistant", response, is_user=False), unsafe_allow_html=True)

# About Section
with st.expander("ℹ️ About AlgoForge"):
    st.markdown(f"""
    **AlgoForge** is your intelligent AI problem solver, tailored for deep tech and coding tasks:
    
    - 🧠 Smart conversational AI with context awareness  
    - 🧑‍💻 Code assistant for explanations, bug fixes, and suggestions  
    - ⚡ Fast response generation using Hugging Face models  
    - 🎯 Streamlined UI for productivity and clarity  

    *Currently powered by: **{selected_model_name}***
    """)
