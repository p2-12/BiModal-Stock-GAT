from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


def corr_graph_topk(returns_window, topk, use_abs=True, add_self_loops=True):
    """Build an undirected top-k correlation graph.

    returns_window: [S, W] (S=stocks, W=window length), typically log returns
    topk: keep top-k neighbors per node (by |corr| if use_abs else corr)

    Returns:
      edge_index: LongTensor [2, E]
      edge_attr:  FloatTensor [E, 2] columns:
        - abs(corr)
        - sign(corr) in {+1,-1}
    """
    S, W=returns_window.shape
    if W<2:
        edge_index=torch.arange(S, dtype=torch.long).repeat(2, 1)
        edge_attr=torch.tensor([[1.0, 1.0]]*S, dtype=torch.float32)
        return edge_index, edge_attr

    C=np.corrcoef(returns_window)
    C=np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    edges=set()
    for i in range(S):
        scores=np.abs(C[i]) if use_abs else C[i]
        order=np.argsort(-scores)
        picks=[j for j in order if j!=i][:topk]
        for j in picks:
            a, b=(i, j) if i<j else (j, i)
            edges.add((a, b))

    edge_src=[]
    edge_dst=[]
    edge_attr=[]
    for a, b in sorted(edges):
        corr=float(C[a, b])
        mag=abs(corr)
        sgn=1.0 if corr>=0 else -1.0
        edge_src+=[a, b]
        edge_dst+=[b, a]
        edge_attr+=[[mag, sgn], [mag, sgn]]

    if add_self_loops:
        for i in range(S):
            edge_src.append(i)
            edge_dst.append(i)
            edge_attr.append([1.0, 1.0])

    edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr


# ------------------------------------------------------------
# EDA + plotting helpers (NetworkX + Matplotlib)
# ------------------------------------------------------------

def corr_matrix(returns_window):
    C=np.corrcoef(returns_window)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def topk_edge_set_from_corr(C, topk, use_abs=True):
    S=C.shape[0]
    edges=set()
    for i in range(S):
        scores=np.abs(C[i]) if use_abs else C[i]
        order=np.argsort(-scores)
        picks=[j for j in order if j!=i][:topk]
        for j in picks:
            a, b=(i, j) if i<j else (j, i)
            edges.add((a, b))
    return edges


def edge_jaccard_similarity(E1, E2):
    if not E1 and not E2:
        return 1.0
    inter=len(E1 & E2)
    union=len(E1 | E2)
    return inter/max(1, union)


def build_nx_graph_from_edge_index(edge_index, edge_attr, tickers=None, sectors=None, drop_self_loops=True):
    G=nx.Graph()

    S=int(edge_index.max().item())+1 if edge_index.numel()>0 else 0
    for i in range(S):
        attrs={}
        if tickers is not None:
            attrs["ticker"]=tickers[i]
        if sectors is not None:
            attrs["sector"]=sectors[i]
        G.add_node(i, **attrs)

    ei=edge_index.detach().cpu().numpy()
    ea=edge_attr.detach().cpu().numpy()
    seen=set()

    for k in range(ei.shape[1]):
        u, v=int(ei[0, k]), int(ei[1, k])
        if drop_self_loops and u==v:
            continue
        a, b=(u, v) if u<v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        abs_corr=float(ea[k, 0])
        sign=float(ea[k, 1])
        G.add_edge(a, b, abs_corr=abs_corr, sign=sign)

    return G


def graph_summary_stats(G):
    n=G.number_of_nodes()
    e=G.number_of_edges()
    density=nx.density(G) if n>1 else 0.0

    comps=list(nx.connected_components(G)) if n>0 else []
    ncomp=len(comps)
    largest=max((len(c) for c in comps), default=0)
    largest_frac=(largest/n) if n>0 else 0.0

    degs=np.array([d for _, d in G.degree()], dtype=float) if n>0 else np.array([0.0])
    mean_deg=float(degs.mean()) if len(degs) else 0.0
    p95_deg=float(np.percentile(degs, 95)) if len(degs) else 0.0

    avg_clust=float(nx.average_clustering(G)) if n>1 else 0.0

    return {
        "n_nodes":float(n),
        "n_edges":float(e),
        "density":float(density),
        "n_components":float(ncomp),
        "largest_component_frac":float(largest_frac),
        "mean_degree":float(mean_deg),
        "p95_degree":float(p95_deg),
        "avg_clustering":float(avg_clust),
    }


def plot_corr_heatmap(returns_window, tickers=None, max_nodes=40, seed=0, title=None):
    S=returns_window.shape[0]
    rng=np.random.default_rng(seed)
    idx=np.arange(S)
    if S>max_nodes:
        idx=rng.choice(idx, size=max_nodes, replace=False)
        idx=np.sort(idx)

    C=corr_matrix(returns_window)
    C=C[np.ix_(idx, idx)]

    plt.figure(figsize=(6, 5))
    plt.imshow(C, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title or "Correlation heatmap (subset)")

    if tickers is not None:
        labels=[tickers[i] for i in idx]
        plt.xticks(range(len(idx)), labels, rotation=90, fontsize=7)
        plt.yticks(range(len(idx)), labels, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def plot_graph_networkx(G, max_nodes=40, seed=0, title=None, node_label="ticker", color_by="sector", edge_width_scale=3.0):
    rng=np.random.default_rng(seed)
    nodes=list(G.nodes())
    if len(nodes)>max_nodes:
        nodes=rng.choice(nodes, size=max_nodes, replace=False).tolist()

    H=G.subgraph(nodes).copy()
    pos=nx.spring_layout(H, seed=seed, k=0.8)

    widths=[]
    for u, v in H.edges():
        w=float(H[u][v].get("abs_corr", 0.0))
        widths.append(edge_width_scale*w)

    if color_by and all(color_by in H.nodes[n] for n in H.nodes()):
        cats=sorted({H.nodes[n][color_by] for n in H.nodes()})
        cat_to_id={c:i for i, c in enumerate(cats)}
        node_colors=[cat_to_id[H.nodes[n][color_by]] for n in H.nodes()]
    else:
        node_colors=0

    if node_label and all(node_label in H.nodes[n] for n in H.nodes()):
        labels={n:H.nodes[n][node_label] for n in H.nodes()}
    else:
        labels=None

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.6)
    nx.draw_networkx_nodes(H, pos, node_size=260, node_color=node_colors, alpha=0.9)
    if labels is not None:
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=7)
    plt.axis("off")
    plt.title(title or "Correlation top-k graph (subset view)")
    plt.tight_layout()
    plt.show()


def plot_degree_distribution(G, title="Degree distribution"):
    degs=np.array([d for _, d in G.degree()], dtype=float)
    plt.figure(figsize=(6, 3))
    plt.hist(degs, bins=30, density=True)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


def graph_stats_over_time_from_returns(returns_tensor, topk, corr_w, step=10, tickers=None, sectors=None, use_abs=True):
    import pandas as pd

    T=returns_tensor.shape[0]
    rows=[]
    for t in range(0, T, step):
        R=returns_tensor[t]
        w=min(corr_w, R.shape[1])
        Rw=R[:, -w:]

        edge_index, edge_attr=corr_graph_topk(Rw, topk=topk, use_abs=use_abs, add_self_loops=False)
        G=build_nx_graph_from_edge_index(edge_index, edge_attr, tickers=tickers, sectors=sectors, drop_self_loops=True)

        stats=graph_summary_stats(G)
        stats["snapshot"]=t
        rows.append(stats)

    return pd.DataFrame(rows)


def edge_jaccard_over_time(returns_tensor, topk, corr_w, step=1, max_pairs=200, use_abs=True):
    T=returns_tensor.shape[0]
    sims=[]
    pairs=0
    prev_edges=None

    for t in range(0, T, step):
        R=returns_tensor[t]
        w=min(corr_w, R.shape[1])
        C=corr_matrix(R[:, -w:])
        edges=topk_edge_set_from_corr(C, topk=topk, use_abs=use_abs)

        if prev_edges is not None:
            sims.append(edge_jaccard_similarity(prev_edges, edges))
            pairs+=1
            if pairs>=max_pairs:
                break

        prev_edges=edges

    return np.array(sims, dtype=float)
