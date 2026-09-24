import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="EDA RAG Assistant",
    page_icon="📚"
)

st.title("📚 EDA RAG Assistant")
st.write("Ask questions about the uploaded PDF")


# -----------------------------
# Load PDF
# -----------------------------
@st.cache_resource
def setup_rag():

    pdf_loader = PyPDFLoader("T20I_rules.pdf")

    documents = pdf_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b"
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question only using the following context.

        If the answer is not present in the context,
        say that the answer is not available in the document.

        Context:
        {context}

        Question:
        {question}
        """
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    return retriever, llm, prompt


# -----------------------------
# Initialize RAG
# -----------------------------
with st.spinner("Loading document and creating embeddings..."):

    retriever, llm, prompt = setup_rag()


# -----------------------------
# Question Input
# -----------------------------
question = st.text_input(
    "Ask a question:",
    placeholder="Example: What is the length of the cricket pitch?"
)


# -----------------------------
# Generate Answer
# -----------------------------
if question:

    with st.spinner("Finding answer..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        chain = prompt | llm

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

    st.subheader("Answer")
    st.write(answer.content)