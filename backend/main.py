from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np
import faiss
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# === Config ===
FEATURE_FILE = "feature_requests.json"
KB_FILE = "kb.json"
SALES_LEADS_FILE = "sales_leads.json"
NEGATIVE_FEEDBACK_FILE = "negative_feedback.json"
UNRESOLVED_TECH_FILE = "unresolved_technical_queries.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Knowledge Base ===
with open(KB_FILE) as f:
    knowledge_base = json.load(f)

kb_texts = [item["topic"] + ". " + item["content"] for item in knowledge_base]

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Mapping ===
num_labels = 3
label_map = {
    0: "Technical Support",
    1: "Product Feature Request",
    2: "Sales Lead"
}

# === Define Intent Classifier with SentenceTransformer ===
class IntentClassifier(nn.Module):
    def __init__(self, sentence_transformer_name, num_labels=3, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.sentence_transformer = SentenceTransformer(sentence_transformer_name)
        embed_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, texts):
        # texts: list of strings
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True).to(device)
        x = self.fc1(embeddings)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# === Load Trained Intent Classifier ===
path  = "./model_training/intent_classifier_sentence_transformers.pt"
intent_model = IntentClassifier(sentence_transformer_name="all-MiniLM-L6-v2", num_labels=num_labels).to(device)
intent_model.load_state_dict(torch.load(path, map_location=device))
intent_model.eval()

# === Semantic Model for KB Search ===
semantic_model = SentenceTransformer("sentence-transformers/gtr-t5-large")

# === Build Semantic Search Index ===
kb_embeddings = semantic_model.encode(kb_texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

# === Pydantic User Query ===
class UserQuery(BaseModel):
    message: str
    context: Optional[Dict] = None

# === Intent Classification Using SentenceTransformer Classifier ===
def classify_intent(message: str) -> str:
    intent_model.eval()
    with torch.no_grad():
        logits = intent_model([message])
        predicted_label = torch.argmax(logits, dim=1).item()
    return label_map[predicted_label]

# === Semantic KB Search ===
def search_kb(message: str):
    query_vec = semantic_model.encode([message], convert_to_numpy=True)
    D, I = index.search(query_vec, k=1)
    best_idx = I[0][0]
    distance = D[0][0]

    if distance < 1.5:
        return knowledge_base[best_idx]["content"]
    return None

# === Extract Sales Lead Info Helper ===
def extract_sales_lead_info(message: str):
    company = None
    team_size = None

    company_match = re.search(r"(?:company|organization|firm|business|corporation|enterprise)[:\s]*([\w\s&]+)", message, re.I)
    if company_match:
        company = company_match.group(1).strip()

    size_match = re.search(r"(\d+)\s*(?:team members|employees|staff|people|persons|members)", message, re.I)
    if size_match:
        team_size = int(size_match.group(1))

    return {
        "company": company,
        "team_size": team_size
    }

# === Response Drafting ===
def generate_response(intent: str, message: str, kb_info: str = ""):
    message_lower = message.lower()

    if intent == "Technical Support":
        if kb_info:
            return f"Thanks for reaching out! Here's something that might help: {kb_info} Does this resolve your issue?"
        else:
            return "Thanks for your query. I couldn't find an immediate answer, but I've routed your request to our technical support team."

    elif intent == "Product Feature Request":
        return f"Thank you for your suggestion! We've logged your feature request: \"{message}\" for our product team to review."

    elif intent == "Sales Lead":
        if not any(word in message_lower for word in ["company", "team", "employees", "organization", "staff"]):
            return "Thanks for your interest! Our sales team will be in touch soon. Could you tell us your company name and team size?"
        else:
            return "Thanks for your interest! Our sales team will be in touch soon."

    return "Thanks for your message. We'll get back to you as soon as possible."

# === Helper to save to a JSON file safely ===
def append_to_json_file(filename: str, data: dict):
    if os.path.exists(filename):
        with open(filename, "r+", encoding="utf-8") as f:
            try:
                current_data = json.load(f)
            except json.JSONDecodeError:
                current_data = []
            current_data.append(data)
            f.seek(0)
            json.dump(current_data, f, indent=2)
            f.truncate()
    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([data], f, indent=2)

# === Main Chat Endpoint ===
@app.post("/chat")
async def handle_chat(query: UserQuery):
    message = query.message
    context = query.context or {}
    sentiment = TextBlob(message).sentiment.polarity
    intent = classify_intent(message)
    escalate = False
    kb_info = None

    lead_info = context.get("lead_info", {"company": None, "team_size": None})

    if intent == "Technical Support":
        kb_info = search_kb(message)
        if not kb_info:
            escalate = True
            # Save unresolved technical query
            unresolved_data = {
                "message": message,
                "sentiment": sentiment,
                "intent": intent
            }
            append_to_json_file(UNRESOLVED_TECH_FILE, unresolved_data)

    elif intent == "Product Feature Request":
        append_to_json_file(FEATURE_FILE, {"feature_request": message})

    elif intent == "Sales Lead":
        extracted_info = extract_sales_lead_info(message)

        if extracted_info["company"]:
            lead_info["company"] = extracted_info["company"]
        if extracted_info["team_size"]:
            lead_info["team_size"] = extracted_info["team_size"]

        if not lead_info["company"] or not lead_info["team_size"]:
            return {
                "intent": intent,
                "response": "Thanks for your interest! Could you tell us your company name and team size so we can better assist you?",
                "escalate": True,
                "context": {"lead_info": lead_info}
            }

        new_lead = {
            "message": message,
            "company": lead_info["company"],
            "team_size": lead_info["team_size"]
        }

        append_to_json_file(SALES_LEADS_FILE, new_lead)

        lead_info = {"company": None, "team_size": None}

    # Flag highly negative sentiment and save feedback
    if sentiment < -0.5:
        escalate = True

        negative_entry = {
            "message": message,
            "sentiment": sentiment,
            "intent": intent
        }
        append_to_json_file(NEGATIVE_FEEDBACK_FILE, negative_entry)

    response = generate_response(intent, message, kb_info)

    return {
        "intent": intent,
        "response": response,
        "escalate": escalate,
        "context": {"lead_info": lead_info} if intent == "Sales Lead" else None
    }
