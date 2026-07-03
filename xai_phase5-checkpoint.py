# =============================================================================
# PHASE 5 — XAI (FAST VERSION)
# TON-IoT  : GNNExplainer
# CIC-IDS  : Gradient-based importance
# Outputs  : CSVs + inline matplotlib plots (no PNG saved)
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch_geometric.nn import GATConv
from torch_geometric.explain import Explainer, GNNExplainer

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ==============================
# PATHS
# ==============================
base_path      = r"C:\Users\sruja\SEM 6\Mini_Project"
graph_path     = os.path.join(base_path, "data", "Graphs")
model_path     = os.path.join(base_path, "models")
output_path    = os.path.join(base_path, "outputs", "xai")
processed_path = os.path.join(base_path, "data", "Processed")
os.makedirs(output_path, exist_ok=True)


# ==============================
# GAT MODEL
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


def load_model(graph, model_file, hidden, heads, device):
    num_classes   = int(graph.y.max().item()) + 1
    node_feat_dim = graph.x.shape[1]
    edge_feat_dim = graph.edge_attr.shape[1]
    model = GAT(in_channels=node_feat_dim, edge_dim=edge_feat_dim,
                hidden=hidden, heads=heads,
                num_classes=num_classes, dropout=0.3).to(device)
    model.load_state_dict(
        torch.load(os.path.join(model_path, model_file), map_location=device))
    model.eval()
    print(f"  Model loaded | classes={num_classes} | "
          f"node_feats={node_feat_dim} | edge_feats={edge_feat_dim}")
    return model


# ==============================
# GNNExplainer (TON-IoT)
# ==============================
def explain_gnnexplainer(graph, model, feature_names,
                          ZERODAY_CLASS, n_explain, gnn_epochs, device):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=gnn_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification",
                          task_level="edge", return_type="raw"),
    )

    rng      = np.random.default_rng(42)
    zd_idx   = torch.where(graph.y == ZERODAY_CLASS)[0]
    seen_idx = torch.where(graph.y != ZERODAY_CLASS)[0]
    zd_sample   = zd_idx[rng.choice(len(zd_idx),
                  min(n_explain, len(zd_idx)), replace=False)].tolist()
    seen_sample = seen_idx[rng.choice(len(seen_idx),
                  min(n_explain, len(seen_idx)), replace=False)].tolist()

    records = []

    def _run(edge_list, label_str):
        print(f"\n  Explaining {len(edge_list)} {label_str} edges "
              f"(GNNExplainer {gnn_epochs} epochs)...")
        for eidx in edge_list:
            eidx     = int(eidx)
            true_cls = int(graph.y[eidx].item())
            try:
                exp      = explainer(x=graph.x, edge_index=graph.edge_index,
                                     edge_attr=graph.edge_attr, index=eidx)
                node_imp = exp.node_mask.mean(0).cpu().numpy() \
                           if exp.node_mask is not None \
                           else np.zeros(graph.x.shape[1])
                emsum    = float(exp.edge_mask.sum().item()) \
                           if exp.edge_mask is not None else 0.0
            except Exception as e:
                print(f"    Edge {eidx}: failed — {e}")
                node_imp = np.zeros(graph.x.shape[1])
                emsum    = 0.0

            top5 = np.argsort(node_imp)[::-1][:5]
            rec  = {"dataset": "", "edge_idx": eidx,
                    "src_node": int(graph.edge_index[0, eidx].item()),
                    "dst_node": int(graph.edge_index[1, eidx].item()),
                    "true_class": true_cls,
                    "is_zeroday": int(true_cls == ZERODAY_CLASS),
                    "edge_mask_sum": round(emsum, 4),
                    "method": "GNNExplainer",
                    "node_importance": node_imp.tolist()}
            for r, i in enumerate(top5, 1):
                rec[f"top{r}_feature"] = feature_names[i] \
                    if i < len(feature_names) else f"feat_{i}"
                rec[f"top{r}_score"]   = round(float(node_imp[i]), 6)
            records.append(rec)
            print(f"    Edge {eidx} ({label_str}) | class={true_cls} | "
                  f"top: {rec['top1_feature']} ({rec['top1_score']:.4f})")

    _run(zd_sample,   "zero-day")
    _run(seen_sample, "seen")
    return records


# ==============================
# Gradient-based (CIC-IDS)
# ==============================
def explain_gradient(graph, model, feature_names,
                     ZERODAY_CLASS, n_explain, device):
    print(f"\n  Using gradient-based importance (fast)...")

    if graph.y.shape[0] == graph.num_edges:
        edge_labels = graph.y
    else:
        edge_labels = graph.y[graph.edge_index[0]]

    rng      = np.random.default_rng(42)
    zd_idx   = torch.where(edge_labels == ZERODAY_CLASS)[0]
    seen_idx = torch.where(edge_labels != ZERODAY_CLASS)[0]
    zd_sample   = zd_idx[rng.choice(len(zd_idx),
                  min(n_explain, len(zd_idx)), replace=False)].tolist()
    seen_sample = seen_idx[rng.choice(len(seen_idx),
                  min(n_explain, len(seen_idx)), replace=False)].tolist()

    records = []

    def _grad_imp(eidx):
        eidx     = int(eidx)
        x_in     = graph.x.clone().detach().requires_grad_(True)
        model.train()
        out      = model(x_in, graph.edge_index, graph.edge_attr)
        true_cls = int(edge_labels[eidx].item())
        out[eidx, true_cls].backward()
        model.eval()
        imp = x_in.grad.abs().mean(0).detach().cpu().numpy()
        return imp, true_cls

    def _run(edge_list, label_str):
        print(f"\n  Explaining {len(edge_list)} {label_str} edges...")
        for eidx in edge_list:
            eidx = int(eidx)
            try:
                node_imp, true_cls = _grad_imp(eidx)
            except Exception as e:
                print(f"    Edge {eidx}: failed — {e}")
                node_imp = np.zeros(graph.x.shape[1])
                true_cls = int(edge_labels[eidx].item())

            top5 = np.argsort(node_imp)[::-1][:5]
            rec  = {"dataset": "", "edge_idx": eidx,
                    "src_node": int(graph.edge_index[0, eidx].item()),
                    "dst_node": int(graph.edge_index[1, eidx].item()),
                    "true_class": true_cls,
                    "is_zeroday": int(true_cls == ZERODAY_CLASS),
                    "edge_mask_sum": 0.0,
                    "method": "Gradient",
                    "node_importance": node_imp.tolist()}
            for r, i in enumerate(top5, 1):
                rec[f"top{r}_feature"] = feature_names[i] \
                    if i < len(feature_names) else f"feat_{i}"
                rec[f"top{r}_score"]   = round(float(node_imp[i]), 6)
            records.append(rec)
            print(f"    Edge {eidx} ({label_str}) | class={true_cls} | "
                  f"top: {rec['top1_feature']} ({rec['top1_score']:.4f})")

    _run(zd_sample,   "zero-day")
    _run(seen_sample, "seen")
    return records


def run_xai(dataset_name, graph_file, model_file, feature_names,
            ZERODAY_CLASS, method="gnnexplainer",
            n_explain=5, gnn_epochs=50, hidden=64, heads=8):

    print("\n" + "="*60)
    print(f"  XAI -- {dataset_name}  [{method}]")
    print("="*60)

    device = torch.device("cpu")
    graph  = torch.load(os.path.join(graph_path, graph_file),
                        weights_only=False).to(device)
    model  = load_model(graph, model_file, hidden, heads, device)

    if method == "gnnexplainer":
        records = explain_gnnexplainer(graph, model, feature_names,
                                       ZERODAY_CLASS, n_explain,
                                       gnn_epochs, device)
    else:
        records = explain_gradient(graph, model, feature_names,
                                   ZERODAY_CLASS, n_explain, device)

    for r in records:
        r["dataset"] = dataset_name

    df = pd.DataFrame(records)
    safe_name = dataset_name.lower().replace("-","_").replace(" ","_")
    csv_out   = os.path.join(output_path, f"{safe_name}_explanations.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n  CSV saved -> {csv_out}")
    return df


# ==============================
# FEATURE NAMES
# ==============================
try:
    _ton = pd.read_csv(os.path.join(processed_path,
                       "toniot_processed.csv"), nrows=0)
    ton_exclude = {"src_ip","dst_ip","type","ts","label"}
    ton_feature_names = [c for c in _ton.columns if c not in ton_exclude]
except Exception:
    ton_feature_names = [f"feat_{i}" for i in range(40)]

try:
    _cic = pd.read_csv(os.path.join(processed_path,
                       "cicids2017_processed.csv"), nrows=0)
    _cic.columns = _cic.columns.str.strip()
    cic_feature_names = [c for c in _cic.columns if c != "Label"]
except Exception:
    cic_feature_names = [f"feat_{i}" for i in range(78)]

print(f"TON-IoT features: {len(ton_feature_names)}")
print(f"CIC-IDS features: {len(cic_feature_names)}")


# ==============================
# RUN XAI
# ==============================
ton_xai = run_xai(
    dataset_name  = "TON-IoT",
    graph_file    = "toniot_graph.pt",
    model_file    = "gat_toniot.pt",
    feature_names = ton_feature_names,
    ZERODAY_CLASS = 4,
    method        = "gnnexplainer",
    n_explain     = 5,
    gnn_epochs    = 50,
    hidden        = 64,
    heads         = 8,
)

cic_xai = run_xai(
    dataset_name  = "CIC-IDS-2017",
    graph_file    = "cicids2017_graph.pt",
    model_file    = "gat_cicids.pt",
    feature_names = cic_feature_names,
    ZERODAY_CLASS = 10,
    method        = "gradient",
    n_explain     = 5,
    hidden        = 64,
    heads         = 8,
)


# ==============================
# INLINE PLOTS — shown directly in notebook
# ==============================

def get_top_features(df, is_zeroday, top_n=8):
    """Aggregate feature importance scores across all explained edges."""
    subset = df[df["is_zeroday"] == is_zeroday]
    feat_scores = {}
    for _, row in subset.iterrows():
        for rank in range(1, 6):
            feat  = row.get(f"top{rank}_feature")
            score = row.get(f"top{rank}_score", 0.0)
            if feat:
                feat_scores[feat] = feat_scores.get(feat, 0.0) + float(score)
    if not feat_scores:
        return [], []
    top = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [t[0] for t in top], [t[1] for t in top]


def get_per_edge_importance(df, feature_names, is_zeroday, top_n=10):
    """Build a matrix of per-edge importance for heatmap."""
    subset = df[df["is_zeroday"] == is_zeroday].reset_index(drop=True)
    if subset.empty:
        return None, None, None

    all_imp = []
    edge_labels = []
    for _, row in subset.iterrows():
        imp = row.get("node_importance", None)
        if imp is not None and len(imp) > 0:
            imp_arr = np.array(imp)
            all_imp.append(imp_arr)
            edge_labels.append(f"Edge {row['edge_idx']}\n(class {row['true_class']})")

    if not all_imp:
        return None, None, None

    mat = np.stack(all_imp)           # [n_edges, n_feats]
    mean_imp = mat.mean(axis=0)
    top_feat_idx = np.argsort(mean_imp)[::-1][:top_n]
    feat_labels  = [feature_names[i] if i < len(feature_names)
                    else f"feat_{i}" for i in top_feat_idx]
    mat_top = mat[:, top_feat_idx]
    return mat_top, feat_labels, edge_labels


# ============================================================
# PLOT 1 — Feature importance bar charts (zero-day vs seen)
#           for both datasets side by side
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Top Feature Importance — Zero-day vs Seen Class",
             fontsize=15, fontweight="bold")

plot_configs = [
    (ton_xai, ton_feature_names, "TON-IoT", 0),
    (cic_xai, cic_feature_names, "CIC-IDS-2017", 1),
]

colors_zd   = "#C0504D"
colors_seen = "#4F81BD"

for df, feat_names, ds_name, row in plot_configs:
    for col, (is_zd, label, color) in enumerate([
            (1, "Zero-day",   colors_zd),
            (0, "Seen class", colors_seen)]):
        ax = axes[row, col]
        names, scores = get_top_features(df, is_zd, top_n=8)
        if not names:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{ds_name} — {label}")
            continue
        y_pos = np.arange(len(names))
        bars  = ax.barh(y_pos, scores[::-1] if len(scores) > 1 else scores,
                        color=color, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n[:22] for n in names[::-1]] if len(names) > 1
                           else [n[:22] for n in names],
                           fontsize=9)
        for bar, val in zip(bars, scores[::-1] if len(scores) > 1 else scores):
            ax.text(bar.get_width() + max(scores)*0.01, bar.get_y() +
                    bar.get_height()/2, f"{val:.5f}",
                    va="center", fontsize=8)
        ax.set_title(f"{ds_name} — {label} edges\n"
                     f"(method: {df['method'].iloc[0]})",
                     fontweight="bold")
        ax.set_xlabel("Cumulative importance score")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, max(scores) * 1.25)

plt.tight_layout()
plt.show()
print("Plot 1 done: Feature importance bar charts")


# ============================================================
# PLOT 2 — Heatmap of per-edge feature importance (TON-IoT)
#          Zero-day edges vs Seen edges
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("TON-IoT — Per-Edge Feature Importance Heatmap (GNNExplainer)",
             fontsize=14, fontweight="bold")

for ax, (is_zd, title) in zip(axes, [
        (1, "Zero-day edges (class 4 = mitm)"),
        (0, "Seen-class edges")]):
    mat, feat_lbls, edge_lbls = get_per_edge_importance(
        ton_xai, ton_feature_names, is_zd, top_n=10)
    if mat is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        continue
    # Normalize each edge row to [0,1] for fair visual comparison
    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    mat_norm = mat / row_max

    sns.heatmap(mat_norm,
                ax=ax,
                xticklabels=[f[:14] for f in feat_lbls],
                yticklabels=edge_lbls,
                cmap="YlOrRd",
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8},
                cbar_kws={"label": "Normalised importance"})
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Top-10 features")
    ax.set_ylabel("Explained edges")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.show()
print("Plot 2 done: TON-IoT heatmap")


# ============================================================
# PLOT 3 — Heatmap of per-edge feature importance (CIC-IDS)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("CIC-IDS-2017 — Per-Edge Feature Importance Heatmap (Gradient)",
             fontsize=14, fontweight="bold")

for ax, (is_zd, title) in zip(axes, [
        (1, "Zero-day edges (class 10)"),
        (0, "Seen-class edges")]):
    mat, feat_lbls, edge_lbls = get_per_edge_importance(
        cic_xai, cic_feature_names, is_zd, top_n=10)
    if mat is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        continue
    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    mat_norm = mat / row_max

    sns.heatmap(mat_norm,
                ax=ax,
                xticklabels=[f[:14] for f in feat_lbls],
                yticklabels=edge_lbls,
                cmap="Blues",
                linewidths=0.5,
                linecolor="white",
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8},
                cbar_kws={"label": "Normalised importance"})
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Top-10 features")
    ax.set_ylabel("Explained edges")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.show()
print("Plot 3 done: CIC-IDS heatmap")


# ============================================================
# PLOT 4 — Edge mask distribution boxplot (TON-IoT only,
#           GNNExplainer produces edge masks; gradient does not)
# ============================================================
ton_zd   = ton_xai[ton_xai["is_zeroday"] == 1]["edge_mask_sum"].tolist()
ton_seen = ton_xai[ton_xai["is_zeroday"] == 0]["edge_mask_sum"].tolist()

if ton_zd or ton_seen:
    fig, ax = plt.subplots(figsize=(7, 5))
    data, labels, palette = [], [], []
    if ton_zd:
        data.append(ton_zd);   labels.extend(["Zero-day"]*len(ton_zd))
        palette.append("#C0504D")
    if ton_seen:
        data.append(ton_seen); labels.extend(["Seen"]*len(ton_seen))
        palette.append("#4F81BD")

    flat_vals  = [v for group in data for v in group]
    flat_lbls  = labels
    plot_df    = pd.DataFrame({"Edge mask sum": flat_vals,
                                "Type": flat_lbls})
    sns.boxplot(data=plot_df, x="Type", y="Edge mask sum",
                palette={"Zero-day": "#C0504D", "Seen": "#4F81BD"},
                width=0.4, linewidth=1.5, ax=ax)
    sns.stripplot(data=plot_df, x="Type", y="Edge mask sum",
                  palette={"Zero-day": "#8B1A1A", "Seen": "#1A3A6B"},
                  size=6, jitter=True, ax=ax, alpha=0.7)
    ax.set_title("TON-IoT — Edge Mask Sum: Zero-day vs Seen\n"
                 "(GNNExplainer subgraph importance)",
                 fontweight="bold")
    ax.set_ylabel("Edge mask sum")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    print("Plot 4 done: Edge mask boxplot")


# ============================================================
# PLOT 5 — Top-1 feature frequency bar chart
#           Which features appear most as top-1 across edges
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Most Frequently Top-Ranked Feature Across Explained Edges",
             fontsize=14, fontweight="bold")

for ax, (df, ds_name) in zip(axes,
        [(ton_xai, "TON-IoT"), (cic_xai, "CIC-IDS-2017")]):
    freq = df.groupby(["top1_feature", "is_zeroday"]).size().unstack(
        fill_value=0).reset_index()
    freq.columns.name = None
    freq = freq.rename(columns={0: "Seen", 1: "Zero-day"})
    freq = freq.set_index("top1_feature")

    colors = []
    if "Zero-day" in freq.columns: colors.append("#C0504D")
    if "Seen"     in freq.columns: colors.append("#4F81BD")

    freq.plot(kind="bar", ax=ax, color=colors,
              edgecolor="white", width=0.6)
    ax.set_title(f"{ds_name}", fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Count of edges where feature ranked #1")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(title="Edge type")
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.show()
print("Plot 5 done: Top-1 feature frequency")


# ============================================================
# PLOT 6 — Score distribution violin plot per dataset
#           Shows spread of top-1 importance scores
# ============================================================
combined = pd.concat([ton_xai, cic_xai], ignore_index=True)
combined["edge_type"] = combined["is_zeroday"].map(
    {1: "Zero-day", 0: "Seen"})

fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=combined, x="dataset", y="top1_score",
               hue="edge_type",
               palette={"Zero-day": "#C0504D", "Seen": "#4F81BD"},
               split=True, inner="quartile",
               linewidth=1.2, ax=ax)
ax.set_title("Distribution of Top-1 Feature Importance Scores\n"
             "Zero-day vs Seen Edges — Both Datasets",
             fontweight="bold")
ax.set_xlabel("Dataset")
ax.set_ylabel("Top-1 importance score")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.show()
print("Plot 6 done: Violin plot")


# ==============================
# SUMMARY
# ==============================
print("\n" + "="*60)
print("PHASE 5 COMPLETE — XAI SUMMARY")
print("="*60)
for name, df in [("TON-IoT", ton_xai), ("CIC-IDS-2017", cic_xai)]:
    zd   = df[df["is_zeroday"] == 1]
    seen = df[df["is_zeroday"] == 0]
    print(f"\n{name}  (method: {df['method'].iloc[0]})")
    if not zd.empty:
        print(f"  Zero-day top feature : "
              f"{zd['top1_feature'].value_counts().idxmax()}")
    if not seen.empty:
        print(f"  Seen-class top feature: "
              f"{seen['top1_feature'].value_counts().idxmax()}")
    print(f"  Edges explained: {len(df)} "
          f"({len(zd)} zero-day, {len(seen)} seen)")

print(f"\nCSVs saved to: {output_path}")