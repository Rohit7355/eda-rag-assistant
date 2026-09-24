from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
#import os
from dotenv import load_dotenv
load_dotenv()

# Load documents
print("Loading documents...")
# Load txt files
#txt_loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
# Load PDFs
pdf_loader = PyPDFLoader("T20I_rules.pdf")

documents = pdf_loader.load()

# Split into chunks
print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)

# Create embeddings and vector store
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Set up LLM and QA chain
print("Setting up LLM...")
llm = ChatGroq(
    model="openai/gpt-oss-120b" ) 
    #api_key=os.getenv("GROQ_API_KEY")


prompt = ChatPromptTemplate.from_template("""
You are an expert assistant.

Answer the question ONLY using the context below.

If the answer exists in the context, provide it clearly.

If the context contains lists or tables, extract the items clearly.

Context:
{context}

Question:
{question}
""")
retriever = vectorstore.as_retriever(search_kwargs={"k":6})

# Interactive Q&A loop
print("\nEDA RAG Assistant ready! Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    print(f"\nAnswer: {answer.content}\n")
