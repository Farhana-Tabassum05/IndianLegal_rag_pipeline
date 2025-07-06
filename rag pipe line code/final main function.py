def legal_rag_pipeline(file_path, user_question):
    global current_pdf_path, final_summary

    is_new_pdf = file_path != current_pdf_path
    if is_new_pdf:
        unload_mythomax()
        pegasus_tokenizer, pegasus_model = load_pegasus()
        chunks = chunk_with_tokenizer(file_path, pegasus_tokenizer)
        store_chunks(chunks)
        top_chunks = retrieve_top_chunks(chunks)
        chunk_summaries = summarize_chunks(top_chunks)
        store_chunk_summaries(chunk_summaries)
        final_summary = rerank_and_generate_final_summary(chunk_summaries)
        store_summary_and_embeddings(final_summary)
        unload_pegasus()
        mytho_tokenizer, mytho_model = load_mythomax()
        current_pdf_path = file_path
    else:
        if mytho_model is None:
            mytho_tokenizer, mytho_model = load_mythomax()

    q_type = detect_question_type(user_question)

    if q_type == "summary":
        return format_for_human(final_summary, mytho_tokenizer, mytho_model)

    elif q_type in ["judgment_reasoning", "case_opinion"]:
        relevant_chunks = retrieve_top_summary_chunks(5)
    else:
        relevant_chunks = retrieve_top_summary_chunks(3)

    return answer_question_from_chunks(relevant_chunks, user_question, q_type, mytho_tokenizer, mytho_model)
