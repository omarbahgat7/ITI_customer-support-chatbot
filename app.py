import os
import joblib
import pickle
import faiss
import torch

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "chatbot_models")

INTENT_DIR = os.path.join(MODELS, "intent")
LANGUAGE_DIR = os.path.join(MODELS, "language")
RAG_DIR = os.path.join(MODELS, "rag")
EMOTION_DIR = os.path.join(MODELS, "emojy")


# ============================================================
# LOAD INTENT
# ============================================================

intent_vectorizer = joblib.load(
    os.path.join(INTENT_DIR, "intent_vectorizer.pkl")
)

intent_classifier = joblib.load(
    os.path.join(INTENT_DIR, "intent_classifier.pkl")
)


# ============================================================
# LOAD LANGUAGE
# ============================================================

language_vectorizer = joblib.load(
    os.path.join(LANGUAGE_DIR, "language_vectorizer.pkl")
)

language_classifier = joblib.load(
    os.path.join(LANGUAGE_DIR, "language_classifier.pkl")
)


# ============================================================
# LOAD RAG
# ============================================================

rag_intent_vectorizer = joblib.load(
    os.path.join(RAG_DIR, "rag_intent_vectorizer.pkl")
)

rag_intent_classifier = joblib.load(
    os.path.join(RAG_DIR, "rag_intent_classifier.pkl")
)

rag_faiss_index = faiss.read_index(
    os.path.join(RAG_DIR, "rag_faiss.index")
)

with open(
    os.path.join(RAG_DIR, "rag_documents.pkl"),
    "rb"
) as f:
    rag_documents = pickle.load(f)


# ============================================================
# LOAD EMOTION
# ============================================================

emotion_tokenizer = AutoTokenizer.from_pretrained(
    EMOTION_DIR
)

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    EMOTION_DIR
)

DEVICE = torch.device("cpu")
emotion_model.to(DEVICE)
emotion_model.eval()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="E-commerce Customer Support Chatbot",
    description="RAG-based NLP customer support chatbot",
    version="1.0"
)


class ChatRequest(BaseModel):
    message: str


# ============================================================
# HELPERS
# ============================================================

def get_document(index):
    """
    Safely retrieve a document from pandas DataFrame,
    list, tuple, or dictionary.
    """

    if hasattr(rag_documents, "iloc"):
        return rag_documents.iloc[index]

    if isinstance(rag_documents, (list, tuple)):
        return rag_documents[index]

    if isinstance(rag_documents, dict):
        return rag_documents[index]

    return None


def retrieve_rag(text, top_k=3):

    # The FAISS index was built using the RAG embedding
    # representation stored by the original pipeline.
    #
    # The original chatbot uses an embedding model before
    # FAISS. Therefore, try to use the same embedding model
    # if available in the package.

    # First classify the RAG intent.
    features = rag_intent_vectorizer.transform([text])

    predicted_intent = rag_intent_classifier.predict(
        features
    )[0]

    results = []

    # --------------------------------------------------------
    # Exact intent filtering
    # --------------------------------------------------------

    if hasattr(rag_documents, "iterrows"):

        for idx, row in rag_documents.iterrows():

            row_intent = row.get("intent")

            if row_intent == predicted_intent:

                response = row.get("response")

                if response:

                    results.append({
                        "score": 1.0,
                        "document": {
                            "instruction": row.get("instruction"),
                            "category": row.get("category"),
                            "intent": row.get("intent"),
                            "response": row.get("response")
                        }
                    })

                    if len(results) >= top_k:
                        break

    else:

        for doc in rag_documents:

            if isinstance(doc, dict):

                if doc.get("intent") == predicted_intent:

                    results.append({
                        "score": 1.0,
                        "document": doc
                    })

                    if len(results) >= top_k:
                        break

    return results


# ============================================================
# CHAT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "E-commerce Customer Support Chatbot"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    text = request.message.strip()

    # ========================================================
    # LANGUAGE
    # ========================================================

    language_features = language_vectorizer.transform(
        [text]
    )

    language = language_classifier.predict(
        language_features
    )[0]


    # ========================================================
    # INTENT
    # ========================================================

    intent_features = intent_vectorizer.transform(
        [text]
    )

    intent = intent_classifier.predict(
        intent_features
    )[0]


    # ========================================================
    # EMOTION
    # ========================================================

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = emotion_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    emotion_id = int(
        torch.argmax(
            probabilities,
            dim=-1
        ).item()
    )

    emotion_confidence = float(
        probabilities[
            0,
            emotion_id
        ].item()
    )

    emotion = emotion_model.config.id2label.get(
        emotion_id,
        f"LABEL_{emotion_id}"
    )


    # ========================================================
    # RAG
    # ========================================================

    rag_results = retrieve_rag(
        text,
        top_k=3
    )


    response = None

    if rag_results:

        best = rag_results[0]

        document = best.get(
            "document",
            {}
        )

        if isinstance(document, dict):

            response = document.get(
                "response"
            )


    # ========================================================
    # FALLBACK
    # ========================================================

    if not response:

        response = (
            "I'm sorry, but I couldn't find a suitable "
            "answer in the support knowledge base."
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "query": text,

        "language": language,

        "intent": intent,

        "emotion": emotion,

        "emotion_confidence":
            emotion_confidence,

        "response": response
    }