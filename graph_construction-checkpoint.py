# =============================================================================
# PHASE 2 — GRAPH CONSTRUCTION
# TON-IoT  → IP-based graph  (nodes = IPs, edges = flows)
# CIC-IDS  → k-NN graph      (nodes = flows, edges = similarity)
# =============================================================================

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os

# ==============================
# PATHS
# ==============================
base_path      = r"C:\Users\sruja\SEM 6\Mini_Project"
processed_path = os.path.join(base_path, "data", "Processed")
graph_path     = os.path.join(base_path, "data", "Graphs")
os.makedirs(graph_path, exist_ok=True)

# ==============================
# LOAD PROCESSED DATA
# ==============================
print("Loading processed CSVs...")
ton_df = pd.read_csv(os.path.join(processed_path, "toniot_processed.csv"))
cic_df = pd.read_csv(os.path.join(processed_path, "cicids2017_processed.csv"))

# Strip spaces from CIC column names
cic_df.columns = cic_df.columns.str.strip()

print(f"TON-IoT shape : {ton_df.shape}")
print(f"CIC-IDS shape : {cic_df.shape}")
print(f"\nTON-IoT columns : {ton_df.columns.tolist()}")
print(f"\nCIC-IDS columns : {cic_df.columns.tolist()}")


# ==============================
# COLUMN DEFINITIONS
# ==============================

# --- TON-IoT ---
ton_src_col      = "src_ip"
ton_dst_col      = "dst_ip"
ton_label_col    = "type"          # multi-class attack type (NOT binary 'label')
ton_exclude      = {ton_src_col, ton_dst_col, ton_label_col, "label", "ts"}
ton_feature_cols = [c for c in ton_df.columns if c not in ton_exclude]

# --- CIC-IDS ---
cic_label_col    = "Label"
cic_exclude      = {cic_label_col}
cic_feature_cols = [c for c in cic_df.columns if c not in cic_exclude]

print(f"\nTON-IoT feature cols ({len(ton_feature_cols)}): {ton_feature_cols}")
print(f"\nCIC-IDS feature cols ({len(cic_feature_cols)}): {cic_feature_cols[:5]} ...")
print(f"\nTON-IoT label distribution:\n{ton_df[ton_label_col].value_counts()}")
print(f"\nCIC-IDS label distribution:\n{cic_df[cic_label_col].value_counts()}")


# ==============================
# HELPER — encode labels to clean int
# ==============================
def encode_labels(series):
    """LabelEncode any label series → 0..N-1 integer array."""
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str))
    print(f"  Label mapping:")
    for i, cls in enumerate(le.classes_):
        count = int((encoded == i).sum())
        print(f"    class_{i} → '{cls}'  ({count} samples)")
    return encoded, le


# ==============================
# TON-IoT: IP-based graph
# Nodes = unique IPs
# Edges = individual flows
# Node features = mean of all flow features touching that IP
# Edge labels = attack type of that flow
# ==============================
def build_ip_graph(df, src_col, dst_col, label_col, feature_cols):
    print(f"\nBuilding IP-based graph from {len(df)} flows...")

    # Encode labels first
    labels_encoded, le = encode_labels(df[label_col])

    all_nodes = pd.concat([df[src_col], df[dst_col]]).unique()
    node_map  = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(node_map)
    print(f"  Unique IPs (nodes): {num_nodes}")

    # Build node feature matrix — average of all flows touching each IP
    node_features = np.zeros((num_nodes, len(feature_cols)))
    counts        = np.zeros(num_nodes)

    feature_vals = df[feature_cols].values.astype(float)
    src_indices  = df[src_col].map(node_map).values
    dst_indices  = df[dst_col].map(node_map).values

    for i in range(len(df)):
        s, d   = src_indices[i], dst_indices[i]
        feats  = feature_vals[i]
        node_features[s] += feats
        node_features[d] += feats
        counts[s] += 1
        counts[d] += 1

    counts = np.maximum(counts, 1).reshape(-1, 1)
    node_features /= counts

    # Replace any remaining NaN/inf
    node_features = np.nan_to_num(node_features, nan=0.0,
                                  posinf=0.0, neginf=0.0)

    edge_index = torch.tensor(
        np.array([src_indices, dst_indices]), dtype=torch.long)
    edge_attr  = torch.tensor(feature_vals, dtype=torch.float)
    y          = torch.tensor(labels_encoded, dtype=torch.long)
    x          = torch.tensor(node_features, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y, num_nodes=num_nodes)

    print(f"  Nodes: {num_nodes} | Edges: {graph.num_edges}")
    print(f"  Node feat dim : {x.shape}")
    print(f"  Edge feat dim : {edge_attr.shape}")
    print(f"  Classes       : {int(y.max().item()) + 1}")
    return graph, le


# ==============================
# CIC-IDS: k-NN similarity graph
# Nodes = individual flows
# Edges = k most similar flows by feature distance
# Node features = flow features
# Edge labels = source node's class
# ==============================
def build_knn_graph(df, label_col, feature_cols, k=5, sample_n=None):
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"  Sampled {sample_n} flows for k-NN graph.")

    print(f"\nBuilding k-NN graph from {len(df)} flows (k={k})...")

    # Encode labels
    labels_encoded, le = encode_labels(df[label_col])

    features = df[feature_cols].values.astype(float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # k-NN edges
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(features)
    distances, indices = nbrs.kneighbors(features)

    src_list, dst_list = [], []
    for node_idx, neighbors in enumerate(indices):
        for nb in neighbors[1:]:          # skip self-loop at index 0
            src_list.append(node_idx)
            dst_list.append(int(nb))

    src_arr = np.array(src_list)
    dst_arr = np.array(dst_list)

    edge_index = torch.tensor(np.array([src_arr, dst_arr]), dtype=torch.long)

    # Edge attr = absolute feature difference between connected flows
    edge_attr  = torch.tensor(
        np.abs(features[src_arr] - features[dst_arr]), dtype=torch.float)

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels_encoded, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y, num_nodes=len(df))

    print(f"  Nodes (flows) : {len(df)} | Edges: {graph.num_edges}")
    print(f"  Node feat dim : {x.shape}")
    print(f"  Edge feat dim : {edge_attr.shape}")
    print(f"  Classes       : {int(y.max().item()) + 1}")
    return graph, le


# ==============================
# BUILD GRAPHS
# ==============================
print("\n" + "="*50)
print("Building TON-IoT graph...")
print("="*50)
ton_graph, le_ton = build_ip_graph(
    ton_df,
    ton_src_col,
    ton_dst_col,
    ton_label_col,
    ton_feature_cols
)

print("\n" + "="*50)
print("Building CIC-IDS-2017 graph...")
print("="*50)
# 200k sample — increase if RAM allows, decrease if it crashes
cic_graph, le_cic = build_knn_graph(
    cic_df,
    cic_label_col,
    cic_feature_cols,
    k=5,
    sample_n=100_000
)


# ==============================
# SAVE
# ==============================
ton_graph_file = os.path.join(graph_path, "toniot_graph.pt")
cic_graph_file = os.path.join(graph_path, "cicids2017_graph.pt")

torch.save(ton_graph, ton_graph_file)
torch.save(cic_graph, cic_graph_file)

print("\n" + "="*50)
print("✅ GRAPHS SAVED")
print("="*50)
print(f"TON-IoT  → {ton_graph_file}")
print(f"CIC-IDS  → {cic_graph_file}")
print(f"\nTON-IoT  graph : {ton_graph}")
print(f"CIC-IDS  graph : {cic_graph}")

# ==============================
# LABEL MAPPING SUMMARY
# (save this — you need ZERODAY_CLASS values for train_gat.py)
# ==============================
print("\n" + "="*50)
print("LABEL MAPPINGS — note these for train_gat.py")
print("="*50)
print("\nTON-IoT  (use one of these as ZERODAY_CLASS):")
for i, cls in enumerate(le_ton.classes_):
    count = int((ton_graph.y == i).sum())
    print(f"  class_{i} → '{cls}'  ({count} edges)")

print("\nCIC-IDS  (use one of these as ZERODAY_CLASS):")
for i, cls in enumerate(le_cic.classes_):
    count = int((cic_graph.y == i).sum())
    print(f"  class_{i} → '{cls}'  ({count} edges)")
