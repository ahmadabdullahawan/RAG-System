import streamlit as st
import os

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="RAG Document Q&A", layout="wide") # MOVED: This line is now at the top
st.title("Retrival Augmentation Generator (RAG) Document Q&A") # MOVED: This line is now at the top
st.markdown("Ask questions about the uploaded PDF documents.")


# --- Install necessary libraries if not already installed ---
# These commands will run at the very start of the Streamlit app execution.
# This ensures all dependencies are installed before any imports are attempted.
# For production deployment, it's generally recommended to manage dependencies
# via a requirements.txt file and install them beforehand.
install_status_placeholder = st.empty() # Placeholder for installation messages
progress_bar_placeholder = st.empty() # Placeholder for progress bar

install_status_placeholder.warning("Ensuring all necessary libraries are installed. This may take a moment...")
progress_bar = progress_bar_placeholder.progress(0)

try:
    # Attempt to import a core library to check if installations are needed
    import langchain
    progress_bar.progress(30)
    import langchain_community
    progress_bar.progress(50)
    import langchain_core
    progress_bar.progress(70)
    import langchain_pinecone
    progress_bar.progress(90)
    import sentence_transformers
    import torch
    import transformers
    import accelerate
    import bitsandbytes
    import pypdf
    import pinecone
    progress_bar.progress(100)
    install_status_placeholder.success("All libraries appear to be installed.")
except ImportError:
    install_status_placeholder.info("Some libraries are missing. Installing them now...")
    os.system("pip install -q langchain pypdf pinecone streamlit langchain-pinecone")
    progress_bar.progress(50)
    os.system("pip install -q sentence-transformers torch transformers accelerate bitsandbytes")
    progress_bar.progress(100)
    install_status_placeholder.success("Libraries installed. Please refresh your browser if issues persist.")
    st.experimental_rerun() # Force a rerun after installation

progress_bar_placeholder.empty() # Clear the progress bar after completion or error
install_status_placeholder.empty() # Clear the installation status message

# Now import after ensuring installation
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
import torch

# --- Configuration ---
# Set Pinecone API Key and Environment as OS environment variables
# Ensure these are set in your environment or directly here for the Streamlit app.
# For deployment, consider using Streamlit Secrets or environment variables.
os.environ["PINECONE_API_KEY"] = "pcsk_4qGfEy_9oT31N1ZdD3qrRHMSEDBHSYpjoJSccrYsBQLxspvSDfHtNx8Dk7T1gcn5Kd7Fr1"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "peaceful-larch" # Your Pinecone index name
DOCUMENT_DIRECTORY = "us_census" # Directory where your PDFs are

# LLM Configuration
LLM_MODEL_NAME = "google/flan-t5-base"
RETRIEVER_K = 5 # Number of documents to retrieve
CHUNK_SIZE = 1000 # Your chosen chunk size
CHUNK_OVERLAP = 200 # Your chosen chunk overlap


# --- Functions to load and cache resources ---

@st.cache_resource
def load_documents_and_populate_pinecone():
    """
    Loads documents, chunks them, embeds them, and populates Pinecone.
    This function will run only once due to st.cache_resource.
    """
    st.info("Loading documents and setting up Pinecone (this may take a moment)...")

    # Create the document directory if it doesn't exist
    # This is important if running in a fresh environment or locally
    os.makedirs(DOCUMENT_DIRECTORY, exist_ok=True)

    # For a real-world scenario, you would place your actual PDF files here.
    # We'll create dummy files if they don't exist, just for initial setup.
    dummy_document_contents = {
        "acsbr-015.pdf": """
        Health Insurance Coverage Status and Type by Geography: 2021 and 2022.
        Issued September 2023.
        Demographic shifts as well as economic and government policy changes can affect people's access to health coverage.
        For example, between 2021 and 2022, the labor market continued to improve, which may have affected private coverage in the United States during that time.
        Public policy changes included the renewal of the Public Health Emergency, which allowed Medicaid enrollees to remain covered under the Continuous Enrollment Provision.
        The American Rescue Plan (ARP) enhanced Marketplace premium subsidies for those with incomes above 400 percent of the poverty level as well as for unemployed people.
        In 2022, the uninsured rate varied from 2.4 percent in Massachusetts to 16.6 percent in Texas.
        Twenty-seven states had lower uninsured rates in 2022 compared with 2021. Maine was the only state whose uninsured rate increased.
        """,
        "acsbr-016.pdf": """
        Poverty in States and Metropolitan Areas: 2022. By Craig Benson. December 2023.
        In 2022, the ACS national poverty rate was 12.6 percent, a decrease from 12.8 percent in 2021.
        The poverty rate decreased in 9 states and the District of Columbia between 2021 and 2022. No state had a poverty rate increase.
        New Hampshire had the lowest 2022 rate at 7.2 percent, while Mississippi and Louisiana had among the highest at 19.1 percent and 18.6 percent, respectively.
        In 5 of the 25 most populous metropolitan areas, the poverty rate decreased between 2021 and 2022.
        The Minneapolis MSA was the only metro area among the 25 most populous metropolitan areas that saw poverty increase.
        """,
        "acsbr-017.pdf": """
        Household Income in States and Metropolitan Areas: 2022. By Kirby G. Posey. December 2023.
        This brief presents statistics on median household income and the Gini index of income inequality.
        The U.S. median household income in 2022 was $74,755, a decline of 0.8 percent from last year, after adjusting for inflation.
        Real median household income increased in five states and decreased in 17 states from 2021 to 2022.
        New Jersey and Maryland had the highest median household incomes of all states. Mississippi had the lowest median household income ($52,719).
        Income inequality in the United States measured by the Gini index increased between 2021 and 2022.
        """,
        "p70-178.pdf": """
        Occupation, Earnings, and Job Characteristics. By Clayton Gumber and Briana Sullivan. July 2022.
        Work is a critical component of our lives and provides a way to obtain material and nonmonetary benefits like employer-provided health insurance.
        Job quality is a multidimensional concept, with considerable disagreement regarding how to best measure it.
        Using data from two surveys administered by the. U.S. Census Bureau, this report highlights common features of employment among the U.S. population,
        including their occupations, work schedules, earnings, and other job characteristics.
        Over 30 percent of workers were employed in just three occupation groups: management (11.3 percent), office and administrative support (10.2 percent), and sales and related (9.5 percent).
        A majority of workers with private health insurance coverage (approximately 86 percent) were covered by an employer-provided health insurance plan.
        """
    }

    for filename, content in dummy_document_contents.items():
        file_path = os.path.join(DOCUMENT_DIRECTORY, filename)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    loader = DirectoryLoader(DOCUMENT_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    st.success(f"Loaded {len(documents)} documents from '{DOCUMENT_DIRECTORY}'.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Split documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    st.success(f"Initialized embedding model: intfloat/multilingual-e5-large")

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    st.success("Initialized Pinecone client.")

    embedding_dimension = embeddings.client.get_sentence_embedding_dimension()

    if INDEX_NAME not in pc.list_indexes().names():
        st.warning(f"Pinecone index '{INDEX_NAME}' does not exist. Creating it...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.success(f"Pinecone index '{INDEX_NAME}' created.")
    else:
        st.info(f"Pinecone index '{INDEX_NAME}' already exists. Connecting to it.")

    pinecone_index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        text_key="page_content",
        namespace="default"
    )

    index_stats = pinecone_index.describe_index_stats()
    current_vector_count = index_stats.namespaces.get('default', {}).get('vector_count', 0)

    if current_vector_count == 0:
        vectorstore.add_documents(chunks)
        st.success(f"Successfully populated Pinecone index '{INDEX_NAME}' with {len(chunks)} chunks.")
    else:
        st.info(f"Pinecone index '{INDEX_NAME}' already contains {current_vector_count} vectors. Skipping re-population.")

    return vectorstore, embeddings

@st.cache_resource
def load_llm_and_chain(_vectorstore_obj, _embeddings_obj): # CHANGED: Added underscore to embeddings_obj
    """
    Loads the LLM and constructs the RAG chain.
    This function will run only once due to st.cache_resource.
    """
    st.info(f"Loading LLM model: {LLM_MODEL_NAME} (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.success(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        load_in_8bit=True if device == "cuda" else False,
        device_map="auto" if device == "cuda" else None
    )
    st.success(f"Loaded LLM model: {LLM_MODEL_NAME}")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("HuggingFacePipeline initialized for LangChain.")

    retriever = _vectorstore_obj.as_retriever(search_kwargs={"k": RETRIEVER_K})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    st.success("RetrievalQA chain constructed.")
    return qa_chain

# Load all components
vectorstore, embeddings = load_documents_and_populate_pinecone()
qa_chain = load_llm_and_chain(vectorstore, embeddings)

# --- User Interface ---
st.header("Ask a Question")
user_query = st.text_input("Enter your question here:", "What was the median household income in the United States in 2022 and what was the change from the previous year?")

if user_query:
    st.subheader("Answer:")
    with st.spinner("Generating answer..."):
        try:
            llm_response = qa_chain.invoke({"query": user_query})
            st.write(llm_response["result"])

            st.subheader("Source Documents:")
            for i, doc in enumerate(llm_response["source_documents"]):
                st.write(f"**Document {i+1} from {doc.metadata.get('source', 'N/A')}**")
                st.info(doc.page_content) # Display full content of the chunk
        except Exception as e:
            st.error(f"An error occurred during Q&A: {e}")
            st.warning("Please ensure the LLM model is fully loaded and you have sufficient resources.")

st.markdown("---")
st.info("Developed with Streamlit, LangChain, Hugging Face, and Pinecone,by Abdullah Ahmad")



