from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# Streamlit 설정
st.set_page_config(
    page_title="DexterBee's AutoMate", page_icon="🐝", initial_sidebar_state="expanded"
)

# 베이스 URL 설정
base_url = st.get_option("server.baseUrlPath")
if base_url:
    st.markdown(f'<base href="{base_url}/">', unsafe_allow_html=True)


load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file")
if uploaded_file:
    st.session_state.uploaded = True

st.title("DexterBee's AutoMate (AI Chatbot) Demo Page")
st.write("---")
message_placeholder = st.empty()
message_placeholder.write(
    "This is a demo version of the AutoMate app. The full version includes more features and better performance, such as an admin page, web search, vector DB management, and more."
)


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


if st.session_state.get("uploaded", False):
    message_placeholder.empty()  # Clear the message if a PDF is uploaded
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # Load into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you with the PDF?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        client = ChatOpenAI(api_key=api_key, model_name="gpt-4o", temperature=0)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Processing..."):
            qa_chain = RetrievalQA.from_chain_type(client, retriever=db.as_retriever())
            result = qa_chain.invoke(prompt)
        msg = result["result"]
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
else:
    st.write("Please upload a PDF to start the chat.")
