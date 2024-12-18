from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
import textwrap
import openai
import streamlit as st
import textwrap

os.environ["OPENAI_API_VERSION"]='OPENAI_API_VERSION'
os.environ["AZURE_OPENAI_API_KEY"]='AZURE_OPENAI_API_KEY'
os.environ["AZURE_OPENAI_ENDPOINT"]='AZURE_OPENAI_ENDPOINT'


# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

# Function to create FAISS database from YouTube video URL
def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(transcript)
    
    # Create and return the FAISS vectorstore
    db = FAISS.from_documents(docs, embeddings)
    return db


# Function to get a response based on a query and the document database
def get_response_from_query(db, query, k=4):
    try:
        # Perform similarity search to find relevant documents
        docs = db.similarity_search(query, k=k)

        # Join the page content of the retrieved documents into a single string
        docs_page_content = " ".join([d.page_content for d in docs])

        # Initialize the AzureChatOpenAI model
        llm = AzureChatOpenAI(model="gpt-35-turbo-16k", temperature=0)

        # Create a prompt template for answering the question
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful assistant that can answer questions about YouTube videos based on the video's transcript.
            Answer the following question: {question}
            By searching the following video transcript: {docs}
            Only use the factual information from the transcript to answer the question.
            If you feel like you don't have enough information to answer the question, say "I don't know."
            Your answers should be verbose and detailed.
            """
        )

        # Set up the chain with the model and prompt
        chain = LLMChain(llm=llm, prompt=prompt)

        # Get the response from the chain
        response = chain.run(question=query, docs=docs_page_content)

        # Clean the response by removing newlines
        response = response.replace("\n", " ")

        return response
    
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

############################streamlitUI#############################################

st.title("YouTube Video Summarizer")

# Sidebar inputs
st.sidebar.header("Input")
video_url = st.sidebar.text_input("YouTube Video URL", "")
query = st.sidebar.text_input("Query", "")

# Process button
if st.sidebar.button("Submit"):
    try:
        st.info("Processing the video and generating response...")

        # Create database from the YouTube video URL
        db = create_db_from_youtube_video_url(video_url)

        # Get response from the query
        response, docs = get_response_from_query(db, query)

        # Display the response
        st.subheader("Response")
        st.write(textwrap.fill(response, width=85))
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
