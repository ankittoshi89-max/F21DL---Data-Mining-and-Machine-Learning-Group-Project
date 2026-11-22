"""
Small demo app for the viva: upload a satellite image and see the predicted land cover.

    streamlit run app/streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MODEL_DIR, CLASSES, OUT_DIR          # noqa: E402
from predict import read_any, to_tiles, CNNPredictor, TabularPredictor, save_map   # noqa: E402

st.set_page_config(page_title="EuroSAT land-cover classifier", layout="wide")
st.title("Land-cover classification from Sentinel-2 imagery")
st.caption("F21DL group project - EuroSAT RGB CNN / 13-band models")

uploaded = st.file_uploader("Upload a JPG/PNG (RGB) or a 13-band GeoTIFF", type=["jpg", "jpeg", "png", "tif", "tiff"])
gsd = st.number_input("Ground resolution of your image (metres per pixel)", value=10.0, min_value=0.5, max_value=100.0)


@st.cache_resource
def get_models(kind):
    found = []
    paths = [MODEL_DIR / "cnn_simple__rgb.pt"] if kind == "rgb" else \
            [MODEL_DIR / "cnn_simple__ms.pt", MODEL_DIR / "random_forest__all.joblib"]
    for p in paths:
        if p.exists():
            found.append(CNNPredictor(p) if p.suffix == ".pt" else TabularPredictor(p))
    return found


if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getbuffer())
    arr, kind = read_any(tmp.name, gsd)
    tiles, grid_shape = to_tiles(arr)
    models = get_models(kind)
    if not models:
        st.error("No trained model found for this input type - train the models first.")
        st.stop()

    for m in models:
        probs = m.predict_proba(tiles)
        st.subheader(f"Model: {m.name}  ({len(tiles)} tile(s))")
        if len(tiles) == 1:
            c1, c2 = st.columns([1, 2])
            if kind == "rgb":
                c1.image(np.clip(arr / 255, 0, 1), caption="input", width=256)
            p = pd.Series(probs[0], index=CLASSES).sort_values(ascending=False)
            c2.metric("Prediction", p.index[0], f"{p.iloc[0] * 100:.1f}% confidence")
            c2.bar_chart(p)
        else:
            out = OUT_DIR / f"app_map_{m.name}.png"
            save_map(arr, kind, probs.argmax(1), grid_shape, out, f"Predicted land cover ({m.name})")
            st.image(str(out))
            shares = pd.Series(probs.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(normalize=True)
            st.dataframe((shares * 100).round(1).rename("% of area"))
