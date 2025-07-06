from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import re

pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(device)


def inject_context(chunk_text: str, case_context: str) -> str:
    return f"[CASE CONTEXT]: {case_context}\n\n{chunk_text}"


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'Indian Kanoon', '', text)
    return text.strip()

def summarize_chunks(chunks, case_context: str, tokenizer, model, batch_size=2):
    model.eval()
    all_summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch = [inject_context(clean_text(ck), case_context) for ck in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        all_summaries.extend(decoded)
    return all_summaries
