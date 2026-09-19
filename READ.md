NLP Final Task 2026
RAG-Based E-commerce Customer Support Chatbot
Project Overview

This project implements an end-to-end NLP-based customer support chatbot for an e-commerce environment.

The system processes each customer query through multiple NLP components:

Language Detection

Emotion Classification

Intent Classification

Retrieval-Augmented Generation (RAG)

FastAPI Deployment

The goal is to provide grounded and contextually appropriate customer-support responses for common topics such as orders, refunds, payments, accounts, and delivery.

System Architecture
Customer Query
      |
      v
Language Detection
      |
      v
Emotion Classification
      |
      v
Intent Classification
      |
      v
Intent Routing
      |
      v
RAG Retrieval
      |
      v
Grounded Response
      |
      v
FastAPI API

1. Language Detection

The language detection module uses traditional NLP techniques with TF-IDF vectorization and a supervised machine-learning classifier.

Dataset

papluca/language-identification

The dataset contains text samples from multiple languages.

Pipeline
Input Text
   ↓
TF-IDF Vectorization
   ↓
LinearSVC Classifier
   ↓
Predicted Language


The detected language is used as part of the overall chatbot analysis.

2. Emotion Classification

The emotion module uses a Transformer-based sequence classification model.

Dataset

dair-ai/emotion

The original dataset contains six emotion classes:

sadness

joy

love

anger

fear

surprise

The trained model is used to identify the emotional tone of the customer message.

Emotion detection is useful for adapting the response and identifying frustrated customers.

3. Intent Classification

The intent classifier is trained using the labeled intent column from the Bitext customer-support dataset.

Approach
Customer Query
      ↓
TF-IDF Vectorization
      ↓
LinearSVC
      ↓
Fine-grained Intent


Examples of supported intents include:

cancel_order

track_order

recover_password

get_refund

check_payment_methods

place_order

change_order

complaint

The predicted intent is also mapped to a higher-level routing category when required.

Example:

cancel_order
      ↓
order_management

4. Retrieval-Augmented Generation (RAG)

The RAG component uses the Bitext customer-support dataset as its knowledge base.

Knowledge Base

The following fields are used:

instruction

response

category

intent

Customer instructions are represented using vector-based retrieval.

Vector Store

FAISS is used as the local vector database.

The project contains:

rag_faiss.index
rag_documents.pkl
rag_intent_vectorizer.pkl
rag_intent_classifier.pkl


The retrieved support response is used as grounded information for answering the customer's question.

This reduces the need for the system to generate unsupported information.

5. Routing

The chatbot uses intent-based routing to determine how a query should be handled.

Examples:

greeting / goodbye / gratitude
        ↓
Direct response

order status
        ↓
RAG

order management
        ↓
RAG

billing and refunds
        ↓
RAG

account management
        ↓
RAG

complaint
        ↓
Priority / escalation handling


Negative or frustrated customer messages can be handled with an empathetic response style.

6. Complete Chatbot Pipeline

For each incoming message, the system performs:

Query
 ↓
Language Detection
 ↓
Intent Classification
 ↓
Emotion Classification
 ↓
RAG Retrieval
 ↓
Final Support Response


Example:

Input:
"I want to cancel my order"

Language:
en

Intent:
cancel_order

Route:
order_management

Emotion:
anger

RAG:
Relevant cancellation-support document

Output:
Grounded cancellation instructions

7. FastAPI Deployment

The chatbot is deployed locally using FastAPI.

The deployment file is:

app.py


The saved models are stored under:

chatbot_models/


Expected project structure:

NLP_Final_Project/
│
├── 01_Language_Detection.ipynb
├── 02_Emotion_Classifier.ipynb
├── 03_Intent_Classifier.ipynb
├── 04_RAG_Pipeline.ipynb
│
├── app.py
│
├── chatbot_models/
│   ├── intent/
│   │   ├── intent_vectorizer.pkl
│   │   └── intent_classifier.pkl
│   │
│   ├── language/
│   │   ├── language_vectorizer.pkl
│   │   └── language_classifier.pkl
│   │
│   ├── rag/
│   │   ├── rag_intent_vectorizer.pkl
│   │   ├── rag_intent_classifier.pkl
│   │   ├── rag_faiss.index
│   │   └── rag_documents.pkl
│   │
│   └── emojy/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── model.safetensors
│
└── README.md

8. Running the API

Install the required dependencies:

pip install fastapi uvicorn scikit-learn joblib faiss-cpu torch transformers pandas


Run the FastAPI application:

uvicorn app:app --host 127.0.0.1 --port 8000


The API will be available locally at:

http://127.0.0.1:8000


FastAPI documentation can be accessed at:

http://127.0.0.1:8000/docs

9. API Endpoints
Health Check
GET /health


Example response:

{
  "status": "healthy"
}

Chat
POST /chat


Request:

{
  "message": "I want to cancel my order"
}


Example response structure:

{
  "query": "I want to cancel my order",
  "language": "en",
  "intent": "cancel_order",
  "emotion": "anger",
  "emotion_confidence": 0.92,
  "response": "..."
}

10. Project Deliverables

The project includes:

Language Detection notebook

Emotion Classification notebook

Intent Classification notebook

RAG Pipeline notebook

Saved trained models

FAISS vector index

FastAPI deployment application

Project documentation

11. Design Decisions
Traditional ML for Language and Intent

TF-IDF with supervised classifiers was selected because these tasks use labeled datasets and traditional NLP provides an efficient and interpretable solution.

Transformer for Emotion Classification

A Transformer-based classifier was selected because emotion classification benefits from contextual text representations.

FAISS for RAG

FAISS was selected as a local vector database because it provides efficient similarity search without requiring an external cloud database.

Local FastAPI Deployment

FastAPI was selected to provide a lightweight local REST API for testing the complete chatbot pipeline.

12. Limitations

The emotion dataset consists of Twitter-style English messages, which differs from real customer-support conversations. Therefore, emotion predictions may not always perfectly represent real customer sentiment.

The RAG responses are grounded in the provided Bitext customer-support knowledge base. If the knowledge base does not contain relevant information, the system may not be able to provide a specific answer.

Conclusion

This project integrates language detection, emotion classification, intent classification, retrieval, and API deployment into a single end-to-end e-commerce customer-support chatbot.

The final system can be executed locally through FastAPI and tested using the automatically generated API documentation.