# =============================================================================
# GAT TRAINING — TON-IoT + CIC-IDS-2017
# Phases 3 & 4: Training, Zero-Day Evaluation, Confidence + Energy Detection
# =============================================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

# ==============================
# PATHS
# ==============================
graph_path = r"C:\Users\sruja\SEM 6\Mini_Project\data\Graphs"
model_path = r"C:\Users\sruja\SEM 6\Mini_Project\models"
os.makedirs(model_path, exist_ok=True)


# ==============================
# GAT MODEL
# 2-layer, 8-head attention, edge features, **kwargs in forward
# ==============================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden=64,
                 heads=8, num_classes=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, hidden, heads=heads,
                             edge_dim=edge_dim, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1,
                             edge_dim=edge_dim, dropout=dropout, concat=False)

        self.bn1 = torch.nn.BatchNorm1d(hidden * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst]], dim=-1)
        return self.edge_mlp(edge_emb)


# ==============================
# ENERGY SCORE HELPER
# ==============================
def energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


# ==============================
# TRAINING LOOP
# ==============================
def train_epoch(model, graph, train_idx_t, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out  = model(graph.x, graph.edge_index, graph.edge_attr)
    loss = criterion(out[train_idx_t], graph.edge_labels[train_idx_t])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, idx_t, criterion):
    model.eval()
    out    = model(graph.x, graph.edge_index, graph.edge_attr)
    preds  = out[idx_t].argmax(dim=1)
    labels = graph.edge_labels[idx_t]
    acc    = (preds == labels).float().mean().item()
    loss   = criterion(out[idx_t], labels).item()
    return loss, acc


# ==============================
# THRESHOLD SWEEP HELPER
# ==============================
def threshold_sweep(test_labels, test_preds, max_conf, zd_mask, ZERODAY_CLASS, num_classes):
    print(f"\n{'Threshold':>10} {'Flagged':>10} {'ZD Recall':>12} "
          f"{'ZD Precision':>14} {'F1 macro':>10}")

    results = []
    for thresh in [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        adj     = test_preds.copy()
        adj[max_conf < thresh] = ZERODAY_CLASS
        flagged = int((max_conf < thresh).sum())
        zd_rec  = float((adj[zd_mask] == ZERODAY_CLASS).mean()) if zd_mask.any() else 0.0
        zd_prec = float(precision_score(
            test_labels, adj, labels=[ZERODAY_CLASS],
            average='macro', zero_division=0))
        f1_mac  = float(f1_score(test_labels, adj, average='macro', zero_division=0))
        print(f"{thresh:>10.2f} {flagged:>10} {zd_rec:>12.4f} "
              f"{zd_prec:>14.4f} {f1_mac:>10.4f}")
        results.append((thresh, flagged, zd_rec, zd_prec, f1_mac))
    return results


# ==============================
# ENERGY SWEEP HELPER
# ==============================
def energy_sweep(test_labels, test_preds, energy_vals,
                 seen_test_mask, zd_mask, ZERODAY_CLASS):
    print(f"\n{'Percentile':>12} {'Threshold':>12} {'Flagged':>10} "
          f"{'ZD Recall':>12} {'F1 macro':>10}")

    for pct in [99, 95, 90, 80, 70, 50]:
        thresh  = float(np.percentile(energy_vals[seen_test_mask], pct))
        ep      = test_preds.copy()
        ep[energy_vals > thresh] = ZERODAY_CLASS
        flagged = int((energy_vals > thresh).sum())
        zd_rec  = float((ep[zd_mask] == ZERODAY_CLASS).mean()) if zd_mask.any() else 0.0
        f1_mac  = float(f1_score(test_labels, ep, average='macro', zero_division=0))
        print(f"{pct:>12} {thresh:>12.4f} {flagged:>10} {zd_rec:>12.4f} {f1_mac:>10.4f}")


# ==============================
# MAIN PIPELINE — runs for one dataset
# ==============================
def run_pipeline(dataset_name, graph_file, model_save_name,
                 ZERODAY_CLASS=None,
                 EPOCHS=100, PATIENCE=15,
                 hidden=64, heads=8, dropout=0.3, lr=1e-3):

    print("\n" + "=" * 60)
    print(f"  DATASET: {dataset_name}")
    print("=" * 60)

    # ----- Load Graph -----
    graph = torch.load(os.path.join(graph_path, graph_file), weights_only=False)

    num_classes   = int(graph.y.max().item()) + 1
    node_feat_dim = graph.x.shape[1]
    print(f"Classes: {num_classes} | Node feat dim: {node_feat_dim}")
    print(f"Label distribution: {dict(zip(*torch.unique(graph.y, return_counts=True)))}")

    # Auto-pick zero-day class (smallest class with enough edges)
    if ZERODAY_CLASS is None:
        counts = torch.bincount(graph.y)
        # pick smallest class that has at least 500 samples
        valid  = [(counts[i].item(), i) for i in range(num_classes)
                  if counts[i].item() >= 500]
        valid.sort()
        ZERODAY_CLASS = valid[0][1]
        print(f"Auto-selected ZERODAY_CLASS = {ZERODAY_CLASS} "
              f"({valid[0][0]} edges)")
    else:
        print(f"ZERODAY_CLASS = {ZERODAY_CLASS}")

    # ----- Split Edges -----
    # REPLACE with this
    all_edge_idx = torch.arange(graph.num_edges)

# TON-IoT: labels are on edges (graph.y has num_edges entries)
# CIC-IDS: labels are on nodes (graph.y has num_nodes entries)
# Detect which case we are in and build edge-level labels accordingly
    if graph.y.shape[0] == graph.num_edges:
    # Edge-level labels (TON-IoT)
        edge_labels = graph.y
    else:
    # Node-level labels (CIC-IDS) — propagate src node label to each edge
        src_nodes   = graph.edge_index[0]
        edge_labels = graph.y[src_nodes]

    seen_mask    = edge_labels != ZERODAY_CLASS
    zeroday_mask = edge_labels == ZERODAY_CLASS

    seen_idx    = all_edge_idx[seen_mask].numpy()
    zeroday_idx = all_edge_idx[zeroday_mask].numpy()

    print(f"Seen edges: {len(seen_idx)} | Zero-day edges: {len(zeroday_idx)}")

    # REPLACE with this
# Check if any class has fewer than 2 samples — if so, skip stratify
    seen_label_counts = np.bincount(edge_labels[seen_idx].numpy())
    use_stratify = bool(np.min(seen_label_counts[seen_label_counts > 0]) >= 2)

    train_idx, temp_idx = train_test_split(
        seen_idx, test_size=0.30, random_state=42,
        stratify=edge_labels[seen_idx].numpy() if use_stratify else None)

    temp_label_counts = np.bincount(edge_labels[temp_idx].numpy())
    use_stratify_temp = bool(np.min(temp_label_counts[temp_label_counts > 0]) >= 2)

    val_idx, test_seen_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42,
        stratify=edge_labels[temp_idx].numpy() if use_stratify_temp else None)

    test_idx = np.concatenate([test_seen_idx, zeroday_idx])
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | "
          f"Test seen: {len(test_seen_idx)} | Test zero-day: {len(zeroday_idx)}")

    # ----- Device & Model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    graph  = graph.to(device)

    graph.edge_labels = edge_labels.to(device)

    model = GAT(
        in_channels=node_feat_dim,
        edge_dim=graph.edge_attr.shape[1],
        hidden=hidden,
        heads=heads,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # Class weights for imbalance
    train_labels  = edge_labels[train_idx]
    class_counts  = torch.bincount(train_labels, minlength=num_classes).float()
    class_counts  = torch.clamp(class_counts, min=1.0)
    class_weights = (class_counts.sum() / (num_classes * class_counts))
    # Cap weights to prevent extreme values from tiny classes
    class_weights = torch.clamp(class_weights, max=10.0).to(device)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx_t   = torch.tensor(val_idx,   dtype=torch.long).to(device)
    test_idx_t  = torch.tensor(test_idx,  dtype=torch.long).to(device)

    # ----- Train -----
    best_val         = float('inf')
    patience_counter = 0
    best_state       = None

    print("\nTraining...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, graph, train_idx_t, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, graph, val_idx_t, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val         = val_loss
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print("✅ Best model restored.")

    # ----- Evaluate -----
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_attr)

    test_labels = graph.edge_labels[test_idx_t].cpu().numpy()
    test_preds  = out[test_idx_t].argmax(dim=1).cpu().numpy()
    test_probs  = F.softmax(out[test_idx_t], dim=1).cpu().numpy()
    max_conf    = test_probs.max(axis=1)

    print("\n" + "=" * 50)
    print(f"PHASE 4 — ZERO-DAY EVALUATION: {dataset_name}")
    print("=" * 50)


    target_names = [f"class_{i}" for i in range(num_classes)]
    if 0 <= ZERODAY_CLASS < num_classes:
        target_names[ZERODAY_CLASS] += " (ZERO-DAY)"
    
    # Only include classes actually present in test set
    present_classes = sorted(np.unique(np.concatenate([test_labels, test_preds])))
    present_names   = [target_names[i] for i in present_classes]
    
    print(classification_report(
        test_labels, test_preds,
        labels=present_classes,
        target_names=present_names,
        zero_division=0))

    # AUC
    try:
        present_classes = np.unique(test_labels)
        if len(present_classes) < 2:
            print("AUC skipped: only one class in test set")
        else:
            # Only score classes that are actually present in test set
            y_bin = label_binarize(test_labels,
                                   classes=list(range(num_classes)))
        # Keep only columns for present classes
            present_mask = np.isin(np.arange(num_classes), present_classes)
            y_bin_present   = y_bin[:, present_mask]
            probs_present   = test_probs[:, present_mask]
        # Renormalize probabilities
            probs_present   = probs_present / probs_present.sum(
                          axis=1, keepdims=True)
            if len(present_classes) == 2:
                auc = roc_auc_score(test_labels,
                                    test_probs[:, present_classes[1]])
            else:
                auc = roc_auc_score(y_bin_present, probs_present,
                                    multi_class='ovr', average='macro')
            print(f"Macro AUC (incl. zero-day): {auc:.4f}")
    except Exception as e:
        print(f"AUC skipped: {e}")


    zd_mask   = test_labels == ZERODAY_CLASS
    zd_recall = float((test_preds[zd_mask] == ZERODAY_CLASS).mean()) \
                if zd_mask.any() else 0.0
    print(f"Zero-day recall (raw classifier): {zd_recall:.4f}  "
          f"({zd_mask.sum()} zero-day edges)")

    # ----- Confidence Threshold Sweep -----
    print("\n--- Confidence threshold sweep ---")
    threshold_sweep(test_labels, test_preds, max_conf,
                    zd_mask, ZERODAY_CLASS, num_classes)

    # Best threshold = 0.90 result
    print("\n--- Best threshold (0.90) detail ---")
    adj_90 = test_preds.copy()
    adj_90[max_conf < 0.90] = ZERODAY_CLASS
    
    present_classes_90 = sorted(np.unique(np.concatenate([test_labels, adj_90])))
    present_names_90   = [target_names[i] for i in present_classes_90]
    
    print(classification_report(
        test_labels, adj_90,
        labels=present_classes_90,
        target_names=present_names_90,
        zero_division=0))
    print(f"Zero-day recall @ 0.90: "
          f"{float((adj_90[zd_mask] == ZERODAY_CLASS).mean()):.4f}")

    # ----- Energy Score Sweep -----
    energy_all = energy_score(out).cpu().numpy()
    energy_test = energy_all[test_idx]
    seen_test_local = test_labels != ZERODAY_CLASS

    print("\n--- Energy score sweep (threshold fitted on seen test edges) ---")
    energy_sweep(test_labels, test_preds, energy_test,
                 seen_test_local, zd_mask, ZERODAY_CLASS)

    # ----- Save Model -----
    torch.save(model.state_dict(),
               os.path.join(model_path, model_save_name))
    print(f"\n✅ Model saved → {model_save_name}")

    # Return model + metadata for XAI
    return {
        "model":        model,
        "graph":        graph,
        "device":       device,
        "num_classes":  num_classes,
        "ZERODAY_CLASS": ZERODAY_CLASS,
        "train_idx":    train_idx,
        "val_idx":      val_idx,
        "test_idx":     test_idx,
        "test_labels":  test_labels,
        "test_preds":   test_preds,
    }


# ==============================
# RUN BOTH DATASETS
# ==============================
if __name__ == "__main__":

    # --- TON-IoT ---
    ton_results = run_pipeline(
        dataset_name    = "TON-IoT",
        graph_file      = "toniot_graph.pt",
        model_save_name = "gat_toniot.pt",
        ZERODAY_CLASS   = 4,   # adjust after checking your label mapping
        EPOCHS          = 200,
        PATIENCE        = 20,
        hidden          = 64,
        heads           = 8,
        dropout         = 0.3,
        lr              = 1e-3,
    )

    # --- CIC-IDS-2017 ---
    cic_results = run_pipeline(
        dataset_name    = "CIC-IDS-2017",
        graph_file      = "cicids2017_graph.pt",
        model_save_name = "gat_cicids.pt",
        ZERODAY_CLASS   = None,  # auto-picks smallest valid class
        EPOCHS          = 150,
        PATIENCE        = 20,
        hidden          = 64,
        heads           = 8,
        dropout         = 0.3,
        lr              = 1e-3,
    )

    print("\n\n✅ TRAINING COMPLETE FOR BOTH DATASETS.")
    print("TON-IoT  model → models/gat_toniot.pt")
    print("CIC-IDS  model → models/gat_cicids.pt")
