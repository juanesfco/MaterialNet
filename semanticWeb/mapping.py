import os
import re
from typing import Dict, Tuple, Iterable, Optional

import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path

CATEGORY_COLORS = {
    "source": "#1f77b4",      # blue
    "function": "#2ca02c",    # green
    "application": "#d62728", # red
}
MODEL_NAME = "all-MiniLM-L6-v2"  # fast & solid SBERT model
SIM_THRESHOLD_DEFAULT = 0.95

def _get_node_category(data: Dict) -> Optional[str]:
    """Robustly read category/type field from node data."""
    for k in ("category", "type", "node_type", "group"):
        if k in data and data[k]:
            return str(data[k]).lower()
    return None


def _get_node_label(nid, data: Dict) -> str:
    """Robustly read a display label for node."""
    for k in ("label", "name", "title"):
        if k in data and data[k]:
            return str(data[k])
    # fallback: use ID as label
    return str(nid)


def _darken_hex(hex_color: str, factor: float = 0.6) -> str:
    """Darken a hex color by factor (0-1, smaller = darker)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(max(0, min(255, r * factor)))
    g = int(max(0, min(255, g * factor)))
    b = int(max(0, min(255, b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())[:60].strip("_") or "query"


def build_mapping_html(
    gexf_path: str,
    category_choice: str,  # "source" or "application"
    query_text: str,
    sim_threshold: float = SIM_THRESHOLD_DEFAULT,
    model_name: str = MODEL_NAME,
    output_html: Optional[str] = None,
    physics: bool = True,
    verbose: bool = False
) -> str:
    """
    Build the mapping for a given query/category and save an interactive HTML.
    Returns the written HTML path.
    """
    category_choice = category_choice.lower().strip()
    if category_choice not in ("source", "application"):
        raise ValueError('category_choice must be "source" or "application".')
    if category_choice == "source":
        category_not_choice = "application"
    else:
        category_not_choice = "source"

    # Load graph
    G = nx.read_gexf(gexf_path)

    # Gather candidate nodes in the selected category
    candidates = []
    for nid, data in G.nodes(data=True):
        cat = _get_node_category(data)
        if cat == category_choice:
            label = _get_node_label(nid, data)
            candidates.append((nid, label))

    if verbose:
        print(f"Amount of candidates: {len(candidates)}.")

    if not candidates:
        raise ValueError(f"No nodes found in category '{category_choice}'.")

    # SBERT embeddings
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query_text], normalize_embeddings=True)
    labels = [lbl for _, lbl in candidates]
    node_embs = model.encode(labels, normalize_embeddings=True)

    # Cosine similarities
    sims = util.cos_sim(query_emb, node_embs).cpu().numpy()[0]  # shape: (n_candidates,)
    matched_idxs = np.where(sims >= sim_threshold)[0].tolist()
    if verbose:
        print(f"Amount of similar candidates: {len(matched_idxs)}.")

    # Always include a “seed” node representing the query itself
    seed_color = _darken_hex(CATEGORY_COLORS[category_choice], factor=0.6)

    # Subset of nodes to visualize
    include_nodes = set()

    # Add matched nodes and their neighbors
    for idx in matched_idxs:
        nid, lbl = candidates[idx]
        include_nodes.add(nid)

        neighs = list(G.neighbors(nid))
        include_nodes.update(neighs)
        
        for nb in neighs:
            neighs_nb = list(G.neighbors(nb))
            for nnb in neighs_nb:
                nnb_cat = _get_node_category(G.nodes[nnb])
                if nnb_cat == category_not_choice:
                    include_nodes.add(nnb)

    # If nothing matched, we still want to show neighbors for the best one, or just the seed alone
    if not include_nodes:
        # pick best match to still show context (optional behavior)
        best_idx = int(np.argmax(sims))
        best_nid, _ = candidates[best_idx]
        include_nodes.add(best_nid)

        neighs = list(G.neighbors(best_nid))
        include_nodes.update(neighs)
        
        for nb in neighs:
            neighs_nb = list(G.neighbors(nb))
            for nnb in neighs_nb:
                nnb_cat = _get_node_category(G.nodes[nnb])
                if nnb_cat == category_not_choice:
                    include_nodes.add(nnb)

    # Build PyVis network
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=True, directed=False, cdn_resources="in_line")
    if physics:
        net.barnes_hut()
    else:
        net.force_atlas_2based()
        #net.toggle_physics(False)

    # Add the seed (query) as its own node (virtual ID)
    seed_id = f"__seed__::{category_choice}::{_slugify(query_text)}"
    net.add_node(
        seed_id,
        label=query_text,
        title=f"<b>Query</b><br>Category: {category_choice}",
        color=seed_color,
        shape="dot",
        size=18,
        borderWidth=2,
    )

    # Add included nodes with category colors and titles
    for nid in include_nodes:
        data = G.nodes[nid]
        cat = _get_node_category(data) or "unknown"
        lbl = _get_node_label(nid, data)

        base_color = CATEGORY_COLORS.get(cat, "#999999")
        title_bits = [f"<b>Label:</b> {lbl}", f"<b>Category:</b> {cat}"]

        # If this node was in the matched set, add similarity to the tooltip
        sim_txt = ""
        if cat == category_choice:
            # find its similarity if applicable
            # we must look it up—candidates contains only chosen category
            try:
                idx = next(i for i, (cid, _) in enumerate(candidates) if cid == nid)
                sim_val = float(sims[idx])
                sim_txt = f"<br><b>Similarity:</b> {sim_val:.3f}"
                title_bits[-1] += sim_txt
            except StopIteration:
                pass

        net.add_node(
            str(nid),
            label=lbl,
            title="<br>".join(title_bits),
            color=base_color,
            shape="dot",
            size=14 if cat == category_choice else 12,
        )

    # Connect seed to all matched nodes in the chosen category to show which ones matched
    for idx in matched_idxs:
        nid, lbl = candidates[idx]
        if str(nid) in net.node_ids:
            net.add_edge(seed_id, str(nid), title="matched ≥ threshold", width=2)

    # Add original edges among included nodes
    for u, v, edata in G.edges(data=True):
        if u in include_nodes and v in include_nodes:
            # simple edge; you can carry weights/titles if present
            net.add_edge(str(u), str(v))

    # Write HTML
    if output_html is None:
        base = os.path.splitext(os.path.basename(gexf_path))[0]
        #output_html = f"mapping_{base}_{category_choice}_{_slugify(query_text)}.html"
        output_html = f"mapping_{base}.html"

    net.show(output_html)

    overlay_html = """
    <div id="legend-box" style="
    position:fixed; right:20px; top:20px;
    background:#fff; border:1px solid #ccc; border-radius:6px;
    padding:10px 12px; font: 13px/1.2 sans-serif; z-index: 999999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
    <div style="font-weight:600; margin-bottom:6px;">Legend</div>
    <div><span style="color:#1f77b4;">&#9679;</span> Source</div>
    <div><span style="color:#2ca02c;">&#9679;</span> Function</div>
    <div><span style="color:#d62728;">&#9679;</span> Application</div>
    </div>
    """

    html_path = Path(output_html)
    html = html_path.read_text(encoding="utf-8")

    # Insert the overlay just before </body>
    html = html.replace("</body>", overlay_html + "\n</body>")

    html_path.write_text(html, encoding="utf-8")
    return output_html