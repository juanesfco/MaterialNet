import os
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import streamlit as st
import streamlit.components.v1 as components

#ROOT = Path(__file__).resolve().parent  
#sys.path.insert(0, str(ROOT))

# ✅ Use your existing function
from semanticWeb.mapping import build_mapping_html

# ---- CONFIG: fixed graph path ----
GEXF_PATH = "semanticWeb/semantic_web.gexf"  # <-- set your fixed path here

st.set_page_config(page_title="MaterialNet", layout="wide")
st.title("MaterialNet")
#st.caption("Uses your existing build_mapping_html() with a fixed .gexf path.")

def _slugify(text: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip())[:60]
    return s.strip("_") or "query"

with st.sidebar:
    st.header("Inputs")
    if not os.path.exists(GEXF_PATH):
        st.error(f"GEXF not found at: {GEXF_PATH}")
        st.stop()

    category_choice = st.radio("Query category", ["source", "application"], index=0)
    query_text = st.text_input("Query text", placeholder="e.g., door")
    threshold = st.slider("Cosine similarity threshold", 0.25, 1.00, value=0.95, step=0.01)
    physics = st.toggle("Enable physics (layout)", value=True)
    verbose = st.toggle("Verbose logs", value=True)

    default_out = "semanticWeb/mapping_app.html"
    output_html = st.text_input("Output HTML path", value=default_out)

    run = st.button("Generate", type="primary", use_container_width=True)

left, right = st.columns([4, 1], gap="large")

if run:
    # Ensure output dir exists
    out_dir = os.path.dirname(output_html) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Call your function and capture stdout if verbose
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                # Try signature with verbose, as in your example
                build_mapping_html(
                    gexf_path=GEXF_PATH,
                    category_choice=category_choice,
                    query_text=query_text,
                    sim_threshold=threshold,
                    output_html=output_html,
                    physics=physics,
                    verbose=verbose,
                )
            except TypeError:
                # Fallback if your function doesn't accept 'verbose'
                build_mapping_html(
                    gexf_path=GEXF_PATH,
                    category_choice=category_choice,
                    query_text=query_text,
                    sim_threshold=threshold,
                    output_html=output_html,
                    physics=physics,
                )
        logs = buf.getvalue()
    except Exception as e:
        st.error(f"Mapping failed: {e}")
        st.stop()

    # Read & display HTML
    try:
        with open(output_html, "r", encoding="utf-8") as f:
            html_str = f.read()
    except Exception as e:
        st.error(f"Could not read output HTML ({output_html}): {e}")
        st.stop()

    with left:
        st.subheader("Network")
        components.html(html_str, height=820, scrolling=True)

    with right:
        st.subheader("Details")
        st.write(f"**GEXF:** `{GEXF_PATH}`")
        st.write(f"**Category:** `{category_choice}`")
        st.write(f"**Query:** `{query_text}`")
        st.write(f"**Threshold:** `{threshold:.2f}`")
        st.write(f"**Physics:** `{physics}`")
        st.write(f"**Output:** `{output_html}`")

        st.download_button(
            "Download HTML",
            data=html_str.encode("utf-8"),
            file_name=os.path.basename(output_html) or "mapping.html",
            mime="text/html",
            use_container_width=True,
        )

        if verbose and logs.strip():
            with st.expander("Logs"):
                st.code(logs)

else:
    st.info("Choose category, enter a query, set the threshold, then click **Generate**.")
    #st.code("streamlit run app.py", language="bash")


#from semanticWeb.mapping import build_mapping_html

#build_mapping_html('semanticWeb/semantic_web.gexf','application','door',sim_threshold=0.75,output_html='semanticWeb/mapping_app.html',physics=False,verbose=True)