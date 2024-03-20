# Import necessary libraries
import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Function to process the text and create a knowledge base
def process_text(text):
    """
    Splits the input text into chunks, generates embeddings for each chunk,
    and stores them in a FAISS vector store.

    Parameters:
    - text (str): The text to be processed.

    Returns:
    - knowledgeBase (FAISS): A FAISS vector store containing the embeddings of the text chunks.
    """
     # Split the text into chunks of size 500 with an overlap of 20 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len
    )
    model_kwargs = {"device": "cuda"}
    chunks = text_splitter.split_text(text)
    # Use HuggingFace embeddings to create embeddings for each chunk
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs=model_kwargs)
    # Create a FAISS vector store from the embeddings
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

# Function to load the Llama model
def load_model(hf_auth):
    """
    Loads the Llama-2 language model with BitsAndBytes quantization for memory optimization.

    Parameters:
    - hf_auth (str): The Hugging Face authentication token.

    Returns:
    - model (transformers.AutoModelForCausalLM): The loaded Llama-2 model.
    - model_id (str): The identifier of the loaded model.
    """
    model_id = 'meta-llama/Llama-2-7b-chat-hf'

    # Configure the BitsAndBytes settings for quantization
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # Load the model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    # Load the model with the specified configuration and quantization settings
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    model.eval()    # Set the model to evaluation mode

    return model, model_id

# Function to train the model on the provided PDF
def train_model(hf_auth, pdf):
    """
    Trains the model on the content of the provided PDF file. It checks for GPU availability,
    loads the Llama-2 model, sets up a tokenizer with stopping criteria, and creates a text
    generation pipeline. It then reads the text from the PDF, processes it into a knowledge base,
    and creates a conversational retrieval chain.

    Parameters:
    - hf_auth (str): The Hugging Face authentication token.
    - pdf (file-like object): The PDF file to be used for training the model.

    Returns:
    - chain (ConversationalRetrievalChain): The conversational retrieval chain created from the model and the knowledge base.
    - status (bool): A flag indicating whether the training was successful.
    """
    with st.sidebar:
        # Check if GPU is available and display the device name
        if torch.cuda.is_available():
            st.write("GPU is available.")
            st.write("Device name:", torch.cuda.get_device_name(0))
        else:
            st.write("GPU is not available.")
    
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    model, model_id = load_model(hf_auth)
    with st.sidebar:
        st.write(f"Model loaded on {device}")

    # Load the tokenizer for the model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    # Define stopping criteria for the generation process
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    # Configure the text generation pipeline
    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    
    # Read the PDF and extract text
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Process the extracted text and create a knowledge base
    vectorstore = process_text(text)
    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    status = True

    return chain, status

def generate_response(prompt_input, chain, chat_history):
    """
    Generates a response to the user's input prompt using the trained conversational retrieval chain.

    Parameters:
    - prompt_input (str): The user's input prompt.
    - chain (ConversationalRetrievalChain): The trained conversational retrieval chain.
    - chat_history (list): A list containing the history of the chat conversation.

    Returns:
    - output (dict): The output generated by the conversational retrieval chain, containing the response to the input prompt.
    """
    output = chain({"question": prompt_input, "chat_history": chat_history})
    return output
