# FILE UPLOAD
FILE_UPLOAD_PATH = "./uploaded_files"

# DATABASE ENDPOINT
QDRANTDB_URL = "http://103.176.178.107:6333"
DB_BATCH_SIZE = 256
DB_TIMEOUT = 60
SAVE_CACHE_FILE_DIR = "dataStorage"

# BM25
ENABLE_BM25 = False
BM25_TOP_K = 100

# EMBEDDING MODEL

# Open AI Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Transformer embedding
# EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
# EMBEDDING_DIM = 768

EMBEDDING_MAX_LENGTH = 256
EMBEDDING_TOP_K = 3

# LLM API ENDPOINT
LLM_MODEL = "ura-hcmut/ura-llama-2.1-8b"
TGI_URL = "https://ws.gvlab.org/fablab/ura/llama/haystack/generate_stream"
API_KEY = ""
MAX_ANSWER_LENGTH = 4096
MAX_MODEL_LENGTH = 8192
REPETITION_PENALTY = 1.0
if "gem" in LLM_MODEL.lower():
    SEPERATORS = "<end_of_turn>\n<start_of_turn>model\n|<end_of_turn>\n<start_of_turn>user\n|<start_of_turn>user\n"
    STOP_WORDS = ["<end_of_turn>"]
elif "mix" in LLM_MODEL.lower():
    SEPERATORS = "\\[\\/INST\\]|<\\/s> \\[INST\\]|<s> \\[INST\\]"
    STOP_WORDS = ["</s>"]
elif "llama" in LLM_MODEL.lower():
    SEPERATORS = "<\\|eot_id\\|>\n<\\|start_header_id\\|>assistant<\\|end_header_id\\|>\n\n|<\\|eot_id\\|>\n<\\|start_header_id\\|>user<\\|end_header_id\\|>\n\n|<\\|start_header_id\\|>user<\\|end_header_id\\|>\n\n"
    STOP_WORDS = ["<|eot_id|>"]
else:
    raise NotImplementedError
    
# FAQ HYPERPARAMETERS
FAQ_FILE = "data/FAQ_testing/queries_and_answers.csv"
FAQ_THRESHOLD = 85
FAQ_TEMPERATURE = 0.6
FAQ_TOP_P = 0.9
FAQ_TOP_K = 50
FAQ_ENABLE_PARAPHRASING = False
FAQ_QUERY_TEMPLATE = """Câu hỏi: {query}
Trả lời: {answer}"""
if "mix" in LLM_MODEL.lower():
    FAQ_PROMPT = """<s> [INST] Bạn là một trợ lý thông minh. Hãy thực hiện các yêu cầu hoặc trả lời câu hỏi từ người dùng bằng tiếng Việt.
Hãy viết lại câu trả lời theo một cách khác dùng thông tin bên dưới.
{query} [/INST]
Câu trả lời mới: """
elif "gem" in LLM_MODEL.lower():
    FAQ_PROMPT = """<start_of_turn>user
Hãy viết lại câu trả lời theo một cách khác dùng thông tin bên dưới.
{query}
<end_of_turn>
<start_of_turn>model
Câu trả lời mới: """
elif "llama"  in LLM_MODEL.lower():
    FAQ_PROMPT = """<|start_header_id|>system<|end_header_id|>\n\nBạn là một trợ lý thông minh. Hãy thực hiện các yêu cầu hoặc trả lời câu hỏi từ người dùng bằng tiếng Việt.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n
Hãy viết lại câu trả lời theo một cách khác dùng thông tin bên dưới.
{query} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n
Câu trả lời mới: """
else:
    raise NotImplementedError

### Bản tin Tuyển dụng số 01-02 (01 - 11/01/2025) 03/01/2025 16:01
# WEB HYPERPARAMETERS
WEB_FILE = "data/URA_Mining/URA-RAG-data.csv"
WEB_THRESHOLD = 0.75 # -> 0.5
WEB_TEMPERATURE = 0.6
WEB_TOP_P = 0.9
WEB_TOP_K = 50
if "mix" in LLM_MODEL.lower():
    WEB_PROMPT = """<s> [INST] Trả lời câu hỏi bằng ngữ cảnh cho sẵn.
Ngữ cảnh: '''
{join(documents, "\n")}
'''
Câu hỏi: {query} [/INST]
Trả lời: """
elif "gem" in LLM_MODEL.lower():
    WEB_PROMPT = """<start_of_turn>user
Trả lời câu hỏi bằng ngữ cảnh cho sẵn.
Ngữ cảnh: '''
{join(documents, "\n")}
'''
Câu hỏi: {query}
<end_of_turn>
<start_of_turn>model
Trả lời: """
elif "llama" in LLM_MODEL.lower():
    WEB_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>---Vai trò---

Bạn là trợ lý hữu ích trả lời câu hỏi của người dùng về Cơ sở tri thức được cung cấp bên dưới.

---Mục tiêu---

Tạo phản hồi ngắn gọn dựa trên Cơ sở tri thức và tuân theo Quy tắc phản hồi.

---Cơ sở tri thức--- '''
{join(documents, "\n")}
'''
---Quy tắc phản hồi---
- Chỉ sử dụng những tri thức cần thiết trong cơ sở tri thức để trả lời.
- Sử dụng định dạng markdown với các sections thích hợp.
- Vui lòng trả lời bằng tiếng Việt.
- Nếu bạn không biết câu trả lời, chỉ cần nói vậy.
- Không bịa đặt ra bất cứ điều gì. Không bao gồm thông tin không được cung cấp bởi Cơ sở tri thức.
- Chỉ trả lời câu hỏi một cách trực tiếp, không viết lại câu truy vấn.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
#     WEB_PROMPT = """
# ---Câu hỏi---
# {query}
# ---Trả lời---
# [Chỉ cung cấp **câu trả lời**]
# """
else:
    raise NotImplementedError
    

FREE_PROMPT = """{query}"""

# OTHERS
WARNING_NOTES = [
    "----------\nXin lưu ý rằng câu trả lời phía trên được sinh ra hoàn toàn từ mô hình ngôn ngữ lớn và không có trong dữ liệu của tôi. Do đó, câu trả lời có thể chứa thông tin sai sự thật hoặc lỗi thời.",
    "----------\nLưu ý: Câu trả lời có thể sai sự thật hoặc lỗi thời do không có trong dữ liệu của chúng tôi.",
]

DEFAULT_ANSWERS = [
    "Xin lỗi! Tôi chưa có dữ liệu về câu hỏi bạn yêu cầu. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Rất tiếc, tôi chưa có câu trả lời cho yêu cầu của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
    "Tôi chưa hiểu câu hỏi của bạn. Vui lòng thử lại hoặc tìm kiếm thông tin trên trang web chính thức của trường: https://hcmut.edu.vn",
]