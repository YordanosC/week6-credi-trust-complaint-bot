# src/prompt_template.py

def build_prompt(context, question):
    template = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer.\n\n"
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return template.format(context=context, question=question)
