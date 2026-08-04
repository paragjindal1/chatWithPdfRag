# ================== STEP 1: LOAD MODULES ==================

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import streamlit as st
import numpy
import time
from PIL import Image
from dotenv import load_dotenv


# ================== STEP 2 API KEYS ==================

st.set_page_config(
    page_title="Chat-With-PDF",
    layout="wide"
)

st.sidebar.title("SET API CONFIG")
st.title("RAG Based Chat With PDF 📑")

GOOGLE_API_KEY = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    type="password"
)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if GOOGLE_API_KEY:
    st.sidebar.info("Api key LOADED")

else:
    st.sidebar.info("Give Api Key")


# =================================== STEP 3: LOAD PDF =====================================

uploaded_file = st.sidebar.file_uploader("upload PDF File", type = ["pdf"])

if uploaded_file:
    with st.spinner("Reading Pdf File"):
        data = uploaded_file.read()
        st.sidebar.pdf(data)

if uploaded_file is not None:
    save_dir = "pdf_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

file_path = os.path.join(save_dir,uploaded_file.name)
with open(file_path,"wb") as f:
    f.write(uploaded_file.getbuffer())
st.write(file_path)


# ================================== Step 4 : LOAD RESOURCES ================================

@st.cache_data
def load_document():
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

#st.cache_data : to load data only one time
# st.cache_resource : to load resorce one time

@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return embeddings


@st.cache_data
def get_splitted_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = splitter.split_documents(documents)
    return chunks

#====================== STEP 5: ===============================

documents = load_document()
embedding = load_embedding()
chunks = get_splitted_chunks()

@st.cache_data
def create_vector_db(chunks,_embeddings):
    vectorstore = FAISS.from_documents(chunks,_embeddings)


    vectorstore.save_local("faiss_index")
    return vectorstore


@st.cache_data
def create_retriever(_vectorstore,k_value):
    retriever = _vectorstore.as_retriever(search_kwargs ={"k":k_value})
    return retriever

vectorstore = create_vector_db(chunks,embedding)
k_slider = st.sidebar("select Top K-Value",min_value = 1,max_value = 10)


retriever = create_retriever(vectorstore,k_slider)


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = ChatPromptTemplate.from_template("""
Answer th question using only the context below,
if the ans is not in context , say i dont know based on the document 

context:{context}
Question:{question}

""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

with st.spinner("Building Rag Chain"):
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )



user_question = st.text_area("Ask Question: ")
if user_question:
    if st.button("Get Answer"):
        with st.spinner("Wait ..."):
            st.write_stream(rag_chain.stream(user_question))


        

