# src/generate_answer.py
from langchain.llms import OpenAI
from src.prompt_template import build_prompt


def generate_answer(question, retrieved_chunks):
    context = "\n---\n".join([doc.page_content for doc in retrieved_chunks])
    prompt = build_prompt(context, question)
    llm = OpenAI(temperature=0.7)  # adjust temp if needed
    return llm.invoke(prompt)
