from transformers import AutoTokenizer, AutoModelForCausalLM
from exllama_hf import ExLlamaHF

def load_mythomax():
    print("🔥 Loading MythoMax + ExLlama")
    tokenizer = AutoTokenizer.from_pretrained("Gryphe/MythoMax-L2-13b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Gryphe/MythoMax-L2-13b",
        trust_remote_code=True,
        quantization_config={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
    )
    model = ExLlamaHF(model)  # Apply ExLlama boost
    return tokenizer, model

mytho_tokenizer, mytho_model = None, None

def unload_pegasus():
    global pegasus_model, pegasus_tokenizer
    del pegasus_model
    del pegasus_tokenizer
    torch.cuda.empty_cache()


def format_for_human(summary: str, mytho_tokenizer, mytho_model):
    prompt = (
        "You are a legal assistant. Reformat the following legal summary to be more human-friendly without adding or hallucinating content."
        f"\n\nSummary:\n{summary}\n\nRewritten Summary:"
    )
    inputs = mytho_tokenizer(prompt, return_tensors="pt").to(mytho_model.device)
    with torch.no_grad():
        outputs = mytho_model.generate(
        **inputs, max_new_tokens=300, do_sample=False)
    return mytho_tokenizer.decode(outputs[0], skip_special_tokens=True)


def answer_question_from_chunks(chunks, question, q_type, mytho_tokenizer, mytho_model):
    context = "\n\n".join(chunks)
    prompt = (
        f"You are a legal assistant. Based on the following legal case information, answer the user's question."
        f"\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = mytho_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(mytho_model.device)
    with torch.no_grad():
        outputs = mytho_model.generate(
        **inputs, max_new_tokens=300, do_sample=False)
    return mytho_tokenizer.decode(outputs[0], skip_special_tokens=True)
