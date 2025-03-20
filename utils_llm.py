import requests
import json
import openai
import numpy as np
from haystack.nodes import BaseRetriever

#----------------------------------------- URA api call -----------------------------------------#

def clean_response(response):
    if "----------" in response:
        response = response.split("----------")[0]
    return response.strip()

def parse_generated_text(lines):
    responses = []

    for raw_data in lines:
        if raw_data.startswith("data:"):
            json_data = json.loads(raw_data[5:])

            if "generated_text" in json_data and json_data["generated_text"]:
                return clean_response(json_data["generated_text"])
            elif "token" in json_data and "text" in json_data["token"]:
                responses.append(json_data["token"]["text"])

    complete_response = " ".join(responses).strip()
    return clean_response(complete_response)

def get_response(prompt):
    url = 'https://ws.gvlab.org/fablab/ura/llama/haystack/generate_stream'
    data = {"inputs": prompt}
    print("PROMPT:\n",prompt)
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        response_text = parse_generated_text(lines)
        return response_text
    else:
        print(f"Lỗi API: {response.status_code}, Nội dung: {response.text}")
        return []
    
#----------------------------------------- OpenAI Retriever -----------------------------------------#

class OpenAIEmbeddingRetriever(BaseRetriever):
    def __init__(self, document_store, api_key, model="text-embedding-3-small", top_k=10):
        super().__init__()
        self.document_store = document_store
        openai.api_key = api_key  # Set API key globally
        self.model = model
        self.top_k = top_k

    def retrieve(self, query, top_k=None, **kwargs):
        """
        Retrieve top_k documents using OpenAI embeddings.
        Supports filters via kwargs.
        """
        top_k = top_k or self.top_k
        query_embedding = self.embed_queries([query])[0]
        return self.document_store.query_by_embedding(query_embedding, top_k=top_k, **kwargs)  # Pass filters

    def retrieve_batch(self, queries, top_k=None, **kwargs):
        """
        Retrieve top_k documents for a batch of queries using OpenAI embeddings.
        Supports filters via kwargs.
        """
        top_k = top_k or self.top_k
        query_embeddings = self.embed_queries(queries)
        return [self.document_store.query_by_embedding(q_emb, top_k=top_k, **kwargs) for q_emb in query_embeddings]

    def embed_queries(self, texts):
        """
        Get OpenAI embeddings for queries.
        """
        response = openai.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = np.array([item.embedding for item in response.data])  # Convert to NumPy array
        return embeddings

    def embed_documents(self, documents):
        """
        Get OpenAI embeddings for documents.
        """
        texts = [doc.content for doc in documents]
        return self.embed_queries(texts)
    
#----------------------------------------- Get Full Prompt Context -----------------------------------------#
def full_ask_prompt_context(query, documents):
    PROMPT_HEAD = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>---Vai trò---
Bạn là trợ lý hữu ích trả lời câu hỏi của người dùng về Cơ sở tri thức được cung cấp bên dưới.
---Mục tiêu---
Tạo phản hồi ngắn gọn dựa trên Cơ sở tri thức và tuân theo Quy tắc phản hồi.
---Cơ sở tri thức---
"""
    PROMPT_TAIL = f"""
---Quy tắc phản hồi---
- Chỉ sử dụng những tri thức cần thiết trong cơ sở tri thức để trả lời.
- Sử dụng định dạng markdown với các sections thích hợp.
- Vui lòng trả lời bằng tiếng Việt.
- Nếu bạn không biết câu trả lời, chỉ cần nói vậy.
- Không bịa đặt ra bất cứ điều gì. Không bao gồm thông tin không được cung cấp bởi Cơ sở tri thức.
- Chỉ trả lời câu hỏi một cách trực tiếp, không viết lại câu truy vấn.
<|start_header_id|>user<|end_header_id|> 
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    documents_string = "\n".join(documents)
    return PROMPT_HEAD + documents_string + PROMPT_TAIL
