import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
# load_dotenv()
# key = os.getenv("GOOGLE_API_KEY")

# for deployment on streamlit
key = st.secrets["GOOGLE_API_KEY"]

# Ensure the API key is loaded
if not key:
    st.error("Google Gemini API key not found. Please create a .env file with GOOGLE_API_KEY.")
    st.stop()

# --- App Configuration ---
APP_TITLE = ":green[Design and Analysis of Algorithms (DAA) Exam Prep]"
SIDEBAR_TITLE = ":green[🤖 DAA Midterm Tutor]"
PROMPT_PLACEHOLDER = "Ask me anything about DAA Midterm Syllabus (Recursion, Complexity, etc.)..."
SUBMIT_BUTTON_LABEL = "Get Answer"
INSTRUCTIONS = """
**Instructions:**

1.  Enter your question in the text area below.
2.  Click the 'Get Answer' button.
3.  The chatbot will provide an answer based on the provided DAA materials.
4.  Focus on topics like:
    * Recursion and its types.
    * Time and Space Complexity analysis.
    * Algorithm design paradigms.
    * Specific algorithms (e.g., sorting, searching).
    * Midterm exam relevant topics.
"""

# --- Setup ---
# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-2.0-flash")

# Load the document
try:
    loader = PyPDFLoader("data.pdf")  # Replace with your PDF
    pages = loader.load_and_split()
except FileNotFoundError:
    st.error("Error: The 'DAA_Midterm_Syllabus.pdf' file was not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the PDF: {e}")
    st.stop()

# Split the document into chunks
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# Create embeddings for the chunks
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")

# Embed each chunk and load it into the vector store
persist_directory = "./chroma_db_daa_midterm_"
db = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
db.persist()

# Connect to the persisted Chroma database
db_connection = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# Create a retriever object from the Chroma database
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Setup chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a specialized AI tutor for Design and Analysis of Algorithms (DAA) mid-term exam preparation.
                                Given a context from the DAA syllabus and a question from the user,
                                you should provide a comprehensive and detailed answer.
                                Focus on explaining concepts clearly, especially for time and space complexity analysis.
                                If the question is about an algorithm's complexity, provide a step-by-step explanation.
                                Summarize the context and explain the concepts in your own words.
                                If the context does not contain the answer, you can state that you don't know."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context: {context}
    Question: {question}
    Answer: """)
])

# Set up output parser
output_parser = StrOutputParser()

# RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(input_dict):
    context = format_docs(input_dict["context"])
    question = input_dict["question"]
    return chat_template.invoke({"context": context, "question": question})

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(generate_response)
    | chat_model
    | output_parser
)

# --- Streamlit UI ---
with st.sidebar:
    st.title(SIDEBAR_TITLE)
    st.markdown("This chatbot is your dedicated tutor for DAA midterm exam preparation.")
    st.markdown(INSTRUCTIONS)

st.title(APP_TITLE)
query = st.text_area("Enter your DAA question:", placeholder=PROMPT_PLACEHOLDER, height=150)

if st.button(SUBMIT_BUTTON_LABEL):
    if query:
        st.subheader("Answer:")
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(query)
            st.write(response)
    else:
        st.warning("Please enter a question.")