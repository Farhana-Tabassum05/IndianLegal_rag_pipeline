current_pdf_path = None
final_summary = None

def unload_mythomax():
    global mytho_model, mytho_tokenizer
    if mytho_model:
        del mytho_model
        del mytho_tokenizer
        torch.cuda.empty_cache()
