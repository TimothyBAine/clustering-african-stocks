"""Matplotlib figures written to ``reports/`` (Agg backend, no display needed)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, leaves_list  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from . import clustering  # noqa: E402


def dendrogram_plot(Z, labels, title, path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.18)))
    dendrogram(Z, labels=list(labels), orientation="left", ax=ax, leaf_font_size=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pca_plot(feats: pd.DataFrame, labels: pd.Series, path, title="") -> None:
    X = clustering.prepare(feats)
    pcs = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels.reindex(feats.index).values, cmap="tab10", s=30)
    for (x, y), name in zip(pcs, feats.index):
        ax.annotate(name.split(":")[1], (x, y), fontsize=6, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="cluster", loc="best")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(df: pd.DataFrame, path, title="", fmt="{:.2f}", vmin=None, vmax=None, cmap="viridis") -> None:
    fig, ax = plt.subplots(figsize=(1.2 * df.shape[1] + 3, 0.6 * df.shape[0] + 2))
    im = ax.imshow(df.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if im.norm(v) < 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coassignment_plot(coassign: pd.DataFrame, Z, path, title="") -> None:
    order = coassign.index[leaves_list(Z)]
    m = coassign.loc[order, order].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "Bootstrap co-assignment (dendrogram order)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of resamples clustered together")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def equity_plot(rets: pd.DataFrame, path, title="", currency="USD") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + rets).cumprod().plot(ax=ax, lw=1.4)
    ax.set_ylabel(f"growth of 1 ({currency})")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_plot(s: pd.Series, path, title="", ylabel="") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    s.plot.bar(ax=ax, color=np.where(s.values >= 0, "#3b7dd8", "#d8613b"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
