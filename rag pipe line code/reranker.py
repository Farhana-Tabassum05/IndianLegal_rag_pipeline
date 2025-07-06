from sentence_transformers import util

def rerank_by_category(summaries, query_templates, embedder):
    summary_embeddings = embedder.encode(summaries, convert_to_tensor=True)
    category_rankings = {}

    for category, query in query_templates.items():
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, summary_embeddings)[0]
        ranked_indices = scores.argsort(descending=True)
        category_rankings[category] = [summaries[i] for i in ranked_indices[:3]]

    return category_rankings

# Example usage:
 templates = {
    "case_context": "What is the case mainly about?",
    "discussion": "What discussions or arguments happened in the case?",
    "final_decision": "What was the final judgment or decision?"
}
 reranked = rerank_by_category(chunk_summaries, templates, retriever_model)
