CrediTrust Complaint Bot
This repository contains the implementation of an AI-powered complaint-answering chatbot for CrediTrust Financial, using Retrieval-Augmented Generation (RAG) to analyze customer complaints across five product categories: Credit Card, Personal Loan, Buy Now Pay Later, Savings Account, and Money Transfers.
Project Structure
credi-trust-complaint-bot/
├── data/
│   ├── raw/                   # CFPB dataset
│   ├── filtered/              # Filtered dataset
│   └── vector_store/          # Vector store for embeddings
├── notebooks/
│   └── 1.0-eda-preprocessing.ipynb
├── src/
│   ├── __init__.py
│   └── text_processing.py
├── requirements.txt
├── .gitignore
└── README.md

Setup Instructions

Clone the repository: git clone https://github.com/yordanosC/week6-credi-trust-complaint-bot.git
Install dependencies: pip install -r requirements.txt
Place the CFPB dataset in data/raw/complaints.csv.
Run the EDA notebook: jupyter notebook notebooks/1.0-eda-preprocessing.ipynb
Run the text processing script: python src/text_processing.py

