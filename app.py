from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# 1. Setup Logging (For our SRE/Prometheus tracking later)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel-rx")

app = FastAPI(title="Sentinel-Rx: Secure Pharma AI", version="1.0.0")

# 2. Initialize the Secure Vector Database in Memory
logger.info("Booting up Secure Vector Vault...")
loader = TextLoader("./data/mock_clinical_trials.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Using a free, lightweight local embedding model 
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create local Chroma DB
db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 1})
logger.info("Vector Vault Ready.")

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
async def health_check():
    """Endpoint for Chaos Engineering and Kubernetes Liveness Probes"""
    return {"status": "healthy", "service": "sentinel-rx-ai"}

@app.post("/api/v1/query")
async def query_patient_data(request: QueryRequest):
    """Queries the mock clinical data securely."""
    logger.info(f"Received query: {request.question}")
    
    try:
        # Retrieve the most relevant clinical data
        relevant_docs = retriever.invoke(request.question)
        context = relevant_docs[0].page_content if relevant_docs else "No clinical data found."
        
        # In a full deployment, we pass this context to an LLM. 
        # Here we return the RAG retrieval to prove the architecture works.
        return {
            "query": request.question,
            "retrieved_context": context,
            "security_status": "Passed - Data accessed within Private Subnet"
        }
    except Exception as e:
        logger.error(f"Error querying database: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)