# Llama 2 Chatbot Trained on Your PDF

This repository contains a Streamlit application that allows users to train a Llama 2 chatbot on a PDF document utilizing LangChain and FAISS vector store and interact with it.

## Overview

The application consists of two main files:

- `app.py`: The Streamlit application file.
- `backend.py`: Contains the backend logic for training the model and generating responses.
  
The chatbot is trained on the text extracted from a PDF file uploaded by the user. The training process involves creating a knowledge base from the text and setting up a conversational retrieval chain. Once trained, the chatbot can generate responses to user inputs based on the knowledge acquired from the PDF.

## Requirements

Before running the application, ensure you have the following requirements installed:

- Python 3.8 or higher
- Streamlit
- Transformers
- Torch
- PyPDF
- FAISS-GPU
- Accelerate
- BitsAndBytes
- LangChain

GPU with CUDA support is also required for the instance to run the application.

## Installation
To install the application on instance with GPU follow these steps:

1. Install and run virtual environment:
```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
2. Install all required python libraries:
```bash
pip install -r requirements.txt
```
3. Run the application:
Move to the folder with `app.py`
```bash
streamlit run app.py
```
If the application has started, you will see a message like:
```text
  You can now view your Streamlit app in your browser.

  Network URL: http://localhost:8501
  External URL: http://<Your IP>:8501
```
Open the link in your browser.

## Usage

This application downloads the initial LLAMA2 model from [Hugging Face](https://huggingface.co/). To download the model, the application needs your Hugging Face access token. You can get it from your [Higging Face Access Token page](https://huggingface.co/settings/tokens). It has the following syntax: "hf_<your_unique_symbols>".

1. Enter Hugging Face Access Token: You can enter it in the sidebar.
2. Upload the PDF file you want to train the chatbot on.
3. Click the "Train Model" button to start the training process. This may take a few minutes, depending on the size of the PDF.
4. Once the training is complete, you can start chatting with the bot. Enter your questions or messages in the chat input box.

## Note
The application is designed for demonstration purposes and may require modifications for production use.

## Acknowledgments

Special thanks to [Meta AI](https://ai.facebook.com/) for developing the [Llama](https://github.com/facebookresearch/llama) model, which powers the chatbot in this application and to [Streamlit](https://streamlit.io) for developing the great framework and comprehensive documentation. 
Their contribution to the field of natural language processing and open-source community are greatly appreciated.
