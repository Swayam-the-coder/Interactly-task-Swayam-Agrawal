import streamlit as st
from streamlit_option_menu import option_menu
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import google.generativeai as genai
import pandas as pd
import os
import bs4

# Set page config
st.set_page_config(page_title='Job-Matching RAG Application', layout='wide', initial_sidebar_state='expanded')

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #E8F6F3;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45A049;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stSidebar>div>div>div>div {
            background-color: #E8F6F3;
        }
        .stFileUploader>label>div>div>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stFileUploader>label>div>div>button:hover {
            background-color: #45A049;
        }
    </style>
""", unsafe_allow_html=True)

# Get API key from the user
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

def text_rag_page():
    st.title('Job-Matching RAG Application 📄')
    
    st.write("### Instructions 📜")
    st.write("""
        1. Enter your Google API Key in the sidebar.
        2. Enter your query in the text input below.
        3. Click the 'Get Results' button to process the text content and provide relevant answers based on the content.
    """)

    def load_and_process_text(file):
        df = pd.read_csv(file)
        content = df.to_csv(index=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(content)
        documents = [Document(page_content=chunk) for chunk in splits]

        if not api_key:
            st.error("Google API Key is required.")
            return None, None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for finding candidates suitable for a job. "
            "Use the csv file containing candidate details to select "
            "a few candidates who are suitable for the job  "
            "based on the given job description "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        question_answer_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain, documents

    file = "RecruterPilot candidate sample input dataset - Sheet1.csv"
    rag_chain = None
    documents = None
    if file:
        rag_chain, documents = load_and_process_text(file)

    input_text = st.text_input("Enter Job Description")

    if st.button("Get Results"):
        if input_text and rag_chain and documents:
            with st.spinner('Processing...'):
                try:
                    response = rag_chain.invoke({"input": input_text, "input_documents": documents})
                    st.write(response["answer"]["output_text"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif input_text:
            st.warning("Please enter a job description.")

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
text_rag_page()


#We are looking for a skilled UI Developer to join our dynamic team. The ideal candidate will have a strong background in front-end development, with proficiency in HTML, CSS, JavaScript, and modern frameworks like React or Angular. Your primary responsibility will be to create visually appealing and user-friendly web interfaces that enhance user experience and align with our brand guidelines
