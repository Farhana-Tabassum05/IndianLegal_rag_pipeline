# installer.py
!pip install -q transformers accelerate sentence-transformers faiss-cpu langchain gTTS pymupdf

# Download Pegasus model and tokenizer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

pegasus_model_name = "google/pegasus-large"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Embedder model for retriever
from sentence_transformers import SentenceTransformer
retriever_model = SentenceTransformer("rossieRuby/NyayaDrishti-bert-v3")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pegasus_model = pegasus_model.to(device)
