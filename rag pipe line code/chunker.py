from transformers import PegasusTokenizer
from typing import List
import math

pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")


def tokenize_and_chunk(text: str, tokenizer, max_tokens: int = 1024) -> List[str]:
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def chunk_pdf_with_tokenizer(text: str) -> List[str]:
    """
    Tokenizer-based chunking using Pegasus tokenizer
    """
    chunks = tokenize_and_chunk(text, pegasus_tokenizer)
    print(f"📄 Total Chunks Created: {len(chunks)}")
    return chunks
