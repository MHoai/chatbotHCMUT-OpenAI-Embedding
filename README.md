# HCMUT Chatbot

## How to install
```
pip install requirements.txt
```

## Reindexing data
The data is stored in the dataStorage folder after indexing. Run development mode if dataStorage already contains files.
```
python main.py --dev --reindex
```

## Run Developing mode
```
python main.py --dev
```

## Code that has been adjusted for OpenAI Embedding.
file envs.py
```python
# Open AI Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
```

file pipelines.py
```python
embedding_retriever = OpenAIEmbeddingRetriever(
document_store=document_store,
api_key=API_KEY,
model=EMBEDDING_MODEL,
top_k=EMBEDDING_TOP_K
)
```

all the code in file utils_llm.py