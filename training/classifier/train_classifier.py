import json
import os
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==== Distributed setup ====
def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# ==== Load data from JSON list with label mapping ====
def load_json_data(path):
    sents1, sents2, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        label_str = entry["label"]
        if label_str == "irrelevant":
            label = 0
        elif label_str == "fuzzy_quote":
            label = 1
        else:
            label = 2
        sents1.append(entry["sentence1"])
        sents2.append(entry["sentence2"])
        labels.append(label)
    return sents1, sents2, labels

# ==== Combine embeddings ====
def combine_embeddings(e1, e2):
    return torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=1)

# ==== Save/load cached embeddings ====
def load_or_compute_embeddings(encoder, sents1, sents2, device):
    if os.path.exists("../../../scratch/tmp/p_krae02/data/embeddings1.pt") and os.path.exists("../../../scratch/tmp/p_krae02/data/embeddings2.pt"):
        emb1 = torch.load("../../../scratch/tmp/p_krae02/data/embeddings1.pt", map_location=device)
        emb2 = torch.load("../../../scratch/tmp/p_krae02/data/embeddings2.pt", map_location=device)
    else:
        emb1 = encoder.encode(sents1, convert_to_tensor=True, device=device,
                              batch_size=64, show_progress_bar=True)
        emb2 = encoder.encode(sents2, convert_to_tensor=True, device=device,
                              batch_size=64, show_progress_bar=True)
        if dist.get_rank() == 0:
            torch.save(emb1, "../../../scratch/tmp/p_krae02/data/embeddings1.pt")
            torch.save(emb2, "../../../scratch/tmp/p_krae02/data/embeddings2.pt")
    return emb1, emb2

# ==== Classifier ====
class PairClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.net(x)

# ==== Main ====
def main():
    local_rank = setup()
    device = torch.device("cuda", local_rank)

    data_path = "../../../scratch/tmp/p_krae02/data/training_classifier.json"
    sents1, sents2, labels = load_json_data(data_path)

    if local_rank == 0:
        print("[Rank 0] Loading or computing sentence embeddings...")
    encoder = SentenceTransformer("../../../scratch/tmp/p_krae02/models/BEmargin_7_0/")
    emb1, emb2 = load_or_compute_embeddings(encoder, sents1, sents2, device)

    X_combined = combine_embeddings(emb1, emb2)
    y_tensor = torch.tensor(labels)

    # ==== Stratified split ====
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined.cpu().numpy(), y_tensor.cpu().numpy(),
        test_size=0.2, stratify=y_tensor.cpu().numpy(), random_state=42
    )

    X_train, X_val = torch.tensor(X_train), torch.tensor(X_val)
    y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=128, sampler=val_sampler)

    input_dim = X_combined.shape[1]
    model = PairClassifier(input_dim).to(device)
    model = DDP(model, device_ids=[local_rank])

    # ==== Class-weighted loss ====
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1, 2]), y=labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    if local_rank == 0:
        print("[Rank 0] Starting training...")
    for epoch in range(1, 51):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss / len(train_loader):.4f}")

    if local_rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        weights_path = f"../../../scratch/tmp/p_krae02/data/classifier_weights_{timestamp}.pt"
        full_path = f"../../../scratch/tmp/p_krae02/data/classifier_full_{timestamp}.pt"
        torch.save(model.module.state_dict(), weights_path)
        torch.save(model.module, full_path)
        print(f" Weights saved to {weights_path}")
        print(f" Full model saved to {full_path}")

        print("\nEvaluating on validation set...")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                y_pred = preds.argmax(dim=1).cpu().numpy()
                all_preds.extend(y_pred)
                all_labels.extend(yb.numpy())

        print(classification_report(all_labels, all_preds, digits=3))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["irrelevant", "fuzzy_quote", "other"])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"logs/confusion_matrix_{timestamp}.png")
        plt.close()
        print(f" Confusion matrix saved to logs/confusion_matrix_{timestamp}.png")


if __name__ == "__main__":
    main()

