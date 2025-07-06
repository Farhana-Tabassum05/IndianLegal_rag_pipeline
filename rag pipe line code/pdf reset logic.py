current_pdf_path = None
final_summary = None

import hashlib

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def unload_mythomax():
    global mytho_model, mytho_tokenizer
    if mytho_model:
        del mytho_model
        del mytho_tokenizer
        torch.cuda.empty_cache()
