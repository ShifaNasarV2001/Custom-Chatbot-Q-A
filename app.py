import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# --- App Configuration ---
st.set_page_config(page_title="Chat with Your Documents", page_icon="📄")
st.title("📄 Chat with Your Documents")
st.markdown("""
Welcome! Upload your PDF documents on the left, click 'Process', and then ask any questions about their content.
""")

# --- Global Settings ---
# Note: Using a temporary directory for the vector store for simplicity in this Streamlit app.
# For a more permanent solution, you might want to manage the path differently.
PERSIST_DIRECTORY = "db"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# --- MODEL CHANGE ---
# Switched from "llama2" to "tinydolphin". This is a much smaller model
# that requires significantly less RAM, which should prevent the memory error.
# Other small models you could try: "phi", "gemma:2b", "tinyllama"
LLM_MODEL = "tinydolphin"

@st.cache_resource
def create_qa_chain(files):
    """
    Processes uploaded PDF files and creates a RetrievalQA chain.
    This function is cached to avoid reprocessing files on every interaction.
    """
    if not files:
        return None

    # Use a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # --- 1. Load Documents ---
        documents = []
        for file in files:
            # Save the uploaded file to the temporary directory
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            
            st.info(f"Loading {file.name}...")
            loader = PyPDFLoader(temp_filepath)
            try:
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
                return None
        
        if not documents:
            st.error("Could not load any documents. Please check the files and try again.")
            return None

        # --- 2. Split Documents into Chunks ---
        st.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        st.success(f"Split into {len(texts)} chunks.")

        # --- 3. Create Vector Store ---
        st.info("Creating vector store... This might take a moment.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        
        # Using a directory for persistence
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        st.success("Vector store created successfully!")

    # --- 4. Initialize QA Chain ---
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    llm = Ollama(model=LLM_MODEL)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF files", type=["pdf"], accept_multiple_files=True
    )
    process_button = st.button("Process Documents")

# --- Main Logic ---
if process_button:
    if uploaded_files:
        with st.spinner("Processing your documents... Please wait."):
            # Create and store the QA chain in the session state
            st.session_state.qa_chain = create_qa_chain(uploaded_files)
        st.success("Documents processed! You can now ask questions.")
    else:
        st.warning("Please upload at least one PDF file.")

# --- Chat Interface ---
st.header("2. Ask Questions")

if "qa_chain" in st.session_state:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your documents?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    
                    # Add source document info to the response
                    sources = "\n\n**Sources:**\n"
                    for doc in result["source_documents"]:
                        source_info = f"- **{doc.metadata.get('source', 'Unknown')}**, page {doc.metadata.get('page', 'N/A')}"
                        sources += f"{source_info}\n"
                    
                    full_response = response + sources
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("Please upload and process your documents to start the chat.")
