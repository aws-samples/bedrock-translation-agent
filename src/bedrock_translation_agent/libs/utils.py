import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_TOKENS_PER_CHUNK = (
    500
)


def split_text_by_token_size(source_text: str, token_limit: int = MAX_TOKENS_PER_CHUNK):
    token_count = num_tokens_in_string(source_text)
    token_size = calculate_chunk_size(
        token_count=token_count, token_limit=token_limit
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=token_size, chunk_overlap=0
    )
    return splitter.split_text(source_text)


def num_tokens_in_string(input_str: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(input_str))


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size
