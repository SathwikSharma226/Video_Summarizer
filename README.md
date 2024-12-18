# Video_Summarizer
This application takes a video link and answers user queries based on its content. Using LLM and LangChain, it helps Bosch Global Software Technologies associates quickly get answers from videos on BoschTube, reducing the time spent watching and summarizing content.

Project Description
This project is an AI-driven YouTube video query application that uses OpenAI's language models and the LangChain framework to provide detailed, query-based answers based on YouTube video content. The application is designed to help users quickly access relevant information from video transcripts without having to watch the entire video. It utilizes advanced techniques such as embeddings, semantic search, and conversational chains to offer intelligent and accurate responses.

Key Features:
YouTube Video Loader: The app allows users to input a YouTube video URL. It automatically fetches and transcribes the video content using the YoutubeLoader from LangChain.
Semantic Search: The transcript is split into smaller chunks, indexed using FAISS, and embedded with Azure OpenAI embeddings. This enables efficient similarity search to retrieve the most relevant sections of the transcript based on the user's query.
Conversational Retrieval Chain: When a user submits a query, the application uses the ConversationalRetrievalChain to search for the most relevant documents in the video transcript and generate a detailed, informative response using an Azure OpenAI model (gpt-35-turbo-16k).
Streamlit UI: A simple and user-friendly interface is built using Streamlit, allowing users to input a YouTube video URL and a query. The application then processes the video and displays the generated answer in real-time.

Technologies Used:

LangChain: A framework for building applications with large language models (LLMs) and document processing pipelines.
Azure OpenAI: For generating embeddings and using GPT models to answer questions.
FAISS: A library for efficient similarity search and clustering of dense vectors.
Streamlit: A framework for creating interactive web applications.

Workflow:

Video Input: The user provides a YouTube video URL in the Streamlit sidebar.
Document Processing: The video’s transcript is fetched and split into smaller chunks for efficient processing and querying.
Vectorization and Indexing: The transcript chunks are embedded using Azure OpenAI embeddings and stored in a FAISS vector database.
Query Processing: The user submits a query, which is used to search for relevant transcript sections using semantic similarity. The results are then passed to a language model for generating a detailed response.
Response Output: The application displays the response to the user in a readable format.

Example Usage:
Input a YouTube video URL in the "YouTube Video URL" field.
Enter a specific query in the "Query" field (e.g., "What is the main topic of the video?").
Click the "Submit" button, and the application will process the video and generate a response based on its transcript.

Installation & Setup:

Install the required dependencies:

pip install -r requirements.txt
Set up your Azure OpenAI environment variables by replacing the placeholders:
python

os.environ["OPENAI_API_VERSION"] = 'your_openai_api_version'
os.environ["AZURE_OPENAI_API_KEY"] = 'your_azure_openai_api_key'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'your_azure_openai_endpoint'
Run the Streamlit application:
streamlit run app.py

Future Enhancements:

Enhanced error handling and edge case management.
Option for users to upload video transcripts directly if YouTube videos are not available.
