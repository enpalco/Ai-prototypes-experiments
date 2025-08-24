# import required packages
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.confluence import ConfluenceReader
import gradio as gr
import re


# setup environment variables
import os
from dotenv import load_dotenv

# Loads variables from .env into your environment
load_dotenv()

# setup up OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 


# Configure async mode for agent
import nest_asyncio
nest_asyncio.apply()


# setup LLM - agent's brain
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding()


# Function to parse pdf path from Gradio input
def extract_pdf_path(query: str): # extract pdf path from query
    path_match = re.search(r"(\S+\.pdf)", query) # regex expression to get pdf path
    if not path_match:
        return "Please include a valid PDF file path in your question."
    pdf_path = path_match.group(1)
    print(pdf_path)
    return pdf_path



# 🎛️ Gradio UI Setup
def chat_with_agent(query): # extract pdf file path from query and run query
    try:
        pdf_path = extract_pdf_path(query) # extract pdf path
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data() # load docs
        splitter = SentenceSplitter(chunk_size=700) # split docs into chunks
        nodes = splitter.get_nodes_from_documents(documents)
        summary_index = SummaryIndex(nodes)
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        response = summary_query_engine.query(query)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


iface = gr.Interface(
    fn=chat_with_agent,
    inputs=gr.Textbox(label="Ask your question", placeholder="e.g., Make sure you include pdf path to the file"),
    outputs="text",
    title="📚 Agent that can do Q&A with PDF file",
    description="Agent Q&A with PDF content (Ai agent prototype - built on LllamaIndex)",
)

# # 🚀 Launch the UI
iface.launch()

