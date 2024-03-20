# Import necessary libraries
import streamlit as st
import os
from backend import * # Import all functions from backend.py
import time

# Configure the layout of the Streamlit page
st.set_page_config(layout="wide")

# Display the title of the app in the center of the page
st.markdown("<h1 style='text-align: center;'>Llama 2 Chatbot trained on your PDF</h1>", unsafe_allow_html=True)

# Function to clear chat history (not used in this version)
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Sidebar configuration
with st.sidebar:
    # Display the title and a link to the model in the sidebar
    st.title('Model')
    st.write('[meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)')

    # Initialize session state variables for tracking progress
    if 'key_success' not in st.session_state:
        st.session_state['key_success'] = False
    if 'pdf_success' not in st.session_state:
        st.session_state['pdf_success'] = False
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False
    ready = st.session_state['ready']

    # Check if Hugging Face API token is provided in secrets
    if 'HF_API_TOKEN' in st.secrets:
        st.success('Access token already provided!', icon='✅')
        hf_auth = st.secrets['HF_API_TOKEN']
        st.session_state['key_success'] = True
    else:
        # Prompt user to enter Hugging Face API token
        hf_auth = st.text_input('Enter Hugging Face access token:', type='password')
        if not (hf_auth.startswith('hf_') and len(hf_auth)==37):
            st.warning('Please enter your credentials!', icon='⚠️')
            st.session_state['key_success'] = False
        else:
            st.success('Proceed to loading PDF file!', icon='👉')
            st.session_state['key_success'] = True
    os.environ['HF_API_TOKEN'] = hf_auth
    
    # Allow user to upload a PDF file if the API token is successfully provided
    if st.session_state['key_success']:
        pdf_file = st.file_uploader("Press Browse to select PDF file", type="pdf", key="pdf_uploader")
        if st.button("Load PDF"):
            if pdf_file is not None:
                st.session_state['pdf_success'] = True
                st.success('PDF file loaded successfully.')
            else:
                st.error("Please upload a PDF file.")
                
    # Enable training the model if the API token and PDF file are successfully provided
    if st.session_state['key_success'] and st.session_state['pdf_success']:
        if st.button('Train model'):
            st.success('Start training model...')
            st.session_state['chain'], st.session_state['ready'] = train_model(hf_auth, pdf_file)
            st.success("Done. Let's chat!")
        

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"sender": "assistant", "text": "How may I assist you today?"}]
    st.session_state.chat_history = []

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["sender"]):
        st.write(message["text"])

#st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input(disabled = not st.session_state['ready']):
    st.session_state.messages.append({"sender": "user", "text": prompt})
    with st.chat_message("sender"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["sender"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, st.session_state['chain'], st.session_state.chat_history)
            #st.session_state.chat_history.append((prompt, response['answer']))
            placeholder = st.empty()
            full_response = ''
            for item in response['answer']:
                full_response += item
                placeholder.markdown(full_response)
                time.sleep(0.02)
            placeholder.markdown(full_response)
    message = {"sender": "assistant", "text": full_response}
    st.session_state.messages.append(message)
