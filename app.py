import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import base64
import PIL.Image as Image

import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import argparse
import streamlit as st

llm_model_name = "llama3"
embeddings_model_name = "all-mpnet-base-v2"  #"all-MiniLM-L6-v2"
persist_directory = "db"
target_source_chunks = 5

def generate_response(user_query):
      
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})  

    llm = Ollama(model=llm_model_name, temperature = 0)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return qa.invoke(user_query)


# app config

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] > .main {
    background-image: url("data:image/png;base64,%s");
    background-position: top-left;
    background-size: 500px 900px;
    background-repeat: no-repeat;
    background-attachment: local;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg("C:\RAG Implementations\BasicRAGChatbot\peakpx.jpg")

user_icon = Image.open('lilo.png')
assistant_icon = Image.open('stitchicon.png')


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hiiii ! How can Stitch help you today?"),
    ]
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar=assistant_icon):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar=user_icon):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar=user_icon):
        st.markdown(user_query)

    with st.chat_message("AI", avatar=assistant_icon):
        AI_response = generate_response(user_query)
        st.write(str(AI_response['result']))

    st.session_state.chat_history.append(AIMessage(content=str(AI_response['result'])))
