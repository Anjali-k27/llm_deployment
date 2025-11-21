import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Quantized LLM Chat",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the quantized model and tokenizer"""
    model_path = "./quantized_model"
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please run quantize.py first to create the quantized model.")
        st.stop()
    
    try:
        with st.spinner("Loading quantized model..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
        st.success("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

def generate_response(tokenizer, model, input_text, max_length=100, temperature=0.7):
    """Generate response from the model"""
    try:
        # Encode input
        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input text from response
        if input_text in response:
            response = response.replace(input_text, "").strip()
        
        return response if response else "I'm not sure how to respond to that."
        
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    st.title("🤖 Quantized LLM Chat Application")
    st.markdown("---")
    
    # Load model
    tokenizer, model = load_model()
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    max_length = st.sidebar.slider("Max Response Length", 50, 200, 100)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    
    # Model info
    st.sidebar.markdown("### Model Info")
    st.sidebar.info("Using quantized DialoGPT-small model for efficient inference")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to talk about?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(
                        tokenizer, model, prompt, max_length, temperature
                    )
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("Actions")
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        st.markdown("### Quick Questions")
        quick_questions = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "What can you help me with?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": question})
                response = generate_response(tokenizer, model, question, max_length, temperature)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Powered by Quantized LLM | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
