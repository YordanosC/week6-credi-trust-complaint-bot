# app.py
import gradio as gr
from src.rag_pipeline import rag_pipeline

def chat_with_rag(user_question):
    answer, sources = rag_pipeline(user_question)
    source_texts = "\n\n".join([doc.page_content for doc in sources])
    return answer, source_texts

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 CrediTrust Complaint Insight Chatbot")
    gr.Markdown("Ask a question about customer complaints across products (e.g., 'Why are people unhappy with BNPL?')")

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question", lines=2, placeholder="Type your question here...")
        ask_button = gr.Button("Ask")

    answer_output = gr.Textbox(label="AI Answer", lines=5)
    sources_output = gr.Textbox(label="Top Retrieved Complaint Excerpts", lines=10)

    ask_button.click(fn=chat_with_rag, inputs=[question_input], outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()
