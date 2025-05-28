import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 3
label_map = {
    "Technical Support": 0,
    "Product Feature Request": 1,
    "Sales Lead": 2
}
batch_size = 16
epochs = 10
learning_rate = 2e-5

# === Load Dataset ===
texts, labels = [], []
with open("customer_intent_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        texts.append(item["text"])
        labels.append(label_map[item["label"]])

print(f"✅ Loaded {len(texts)} examples")

# === Dataset class ===
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = IntentDataset(X_train, y_train)
val_dataset = IntentDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Model Definition ===
# Load base SentenceTransformer model (pretrained transformer + pooling)
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Freeze transformer parameters optionally, here we fine-tune all
# for param in base_model.parameters():
#     param.requires_grad = True

# Add classification head on top of SentenceTransformer output
import torch.nn.functional as F
class IntentClassifier(nn.Module):
    def __init__(self, sentence_transformer, num_labels, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        self.sentence_transformer = sentence_transformer
        embed_dim = sentence_transformer.get_sentence_embedding_dimension()
        
        # Two-layer feedforward classifier head
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, texts):
        # Get sentence embeddings (batch of strings)
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True).to(device)
        
        x = self.fc1(embeddings)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

model = IntentClassifier(base_model, num_labels).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# === Training Loop ===
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()
        logits = model(texts_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts_batch, labels_batch in val_loader:
            labels_batch = labels_batch.to(device)
            logits = model(texts_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1} Validation Weighted F1: {val_f1:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "intent_classifier_sentence_transformers.pt")
print("🎉 Saved fine-tuned model to intent_classifier_sentence_transformers.pt")
