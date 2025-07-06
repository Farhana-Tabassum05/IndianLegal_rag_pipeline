def compose_final_summary(reranked_summary_dict: dict) -> str:
    """
    Join reranked summaries into a final human-friendly summary with grounding.
    """
    sections = {
        "case_context": "🧾 Case Overview:",
        "discussion": "🧠 Key Discussions:",
        "final_decision": "⚖️ Final Decision:"
    }
    
    full_summary = "\n\n".join(
        f"{sections[key]}\n{chr(10).join(summaries)}" for key, summaries in reranked_summary_dict.items()
    )
    return full_summary
