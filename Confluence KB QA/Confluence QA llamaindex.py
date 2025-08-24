# import required packages
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.confluence import ConfluenceReader
import gradio as gr


# setup environment variables
import os
from dotenv import load_dotenv

# Loads variables from .env into your environment
load_dotenv()

#from google.colab import userdata
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#GROQ_API_KEY = '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'
#os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Configure async mode for agent
import nest_asyncio
nest_asyncio.apply()

# setup LLM - agent's brain
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding()


# Load Confluence pages as documents
reader = ConfluenceReader(
    base_url="https://albertcortez.atlassian.net/wiki",
    #user_name= userdata.get('Confluence_username')
    user_name = os.getenv('Confluence_username'),
    password = os.getenv('CONFLUENCE_API_TOKEN'),    
    #password= userdata.get('CONFLUENCE_API_TOKEN') # fixed by mapping api token to 'password' instead of api_token
)

# Load all pages from your space
documents = reader.load_data(space_key="~7120209c57022651434e618f4e61a873cf0157")  # limit controls how many pages to fetch


# Split documents into chunks
splitter = SentenceSplitter(chunk_size=700)
nodes = splitter.get_nodes_from_documents(documents)


# Setup search index for chunked docs
summary_index = SummaryIndex(nodes)

# Setup query engine using summary index
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)


# 🎛️ Gradio UI Setup
def chat_with_agent(query):
    try:
        response = summary_query_engine.query(query)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


iface = gr.Interface(
    fn=chat_with_agent,
    inputs=gr.Textbox(label="Ask your question", placeholder="e.g., What are benefits of Charging and Billing Evolved in bullet points"),
    outputs="text",
    title="📚 Agent that can do Q&A with Confluence space content",
    description="Confluence Knowledge base (Ai agent prototype - built on LllamaIndex)",
)

# # 🚀 Launch the UI
iface.launch()



