"""LeafCheck - Plant Disease Detector (Streamlit app)"""
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
HISTORY_PATH = BASE_DIR / "models" / "history.json"
SAMPLES_DIR = BASE_DIR / "sample_images"
IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="LeafCheck - Plant Disease Detector", page_icon="🌿", layout="wide")

# ------------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Figtree:wght@400;500;600&display=swap');
:root{
  --leaf:#1E5B3A; --sage:#F3F6EF; --ink:#18261E; --muted:#5B6B60;
  --ok:#2E7D4F; --bad:#B5452A; --unsure:#C98A1B; --line:#D5DECF; --mint:#E9F0E3;
}
html, body, .stApp { font-family:'Figtree', sans-serif; color:var(--ink); }
.stApp { background:var(--sage); }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2rem; max-width:1150px; }

.lc-hero { background:var(--leaf); border-radius:24px; padding:2.4rem 2.6rem; margin-bottom:1.6rem; }
.lc-hero h1 { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:clamp(1.9rem,4.2vw,3.1rem);
  line-height:1.05; letter-spacing:-0.02em; color:#F3F6EF; margin:0 0 .7rem 0; padding:0; }
.lc-hero p { max-width:54ch; margin:0; font-size:1.06rem; line-height:1.5; color:#D6E6D2; }

.lc-slip { background:#fff; border:1px solid var(--line); border-left-width:9px; border-radius:6px 18px 18px 6px;
  padding:1.3rem 1.5rem 1.2rem; }
.lc-slip.ok { border-left-color:var(--ok); }
.lc-slip.bad { border-left-color:var(--bad); }
.lc-slip.unsure { border-left-color:var(--unsure); }
.lc-plant { font-size:.98rem; color:var(--muted); font-weight:500; }
.lc-condition { font-family:'Bricolage Grotesque', sans-serif; font-weight:700; font-size:2.05rem; line-height:1.1;
  letter-spacing:-0.01em; color:var(--ink); margin:.2rem 0 .7rem; }
.lc-pill { display:inline-block; padding:.18rem .75rem; border-radius:999px; font-size:.86rem; font-weight:600; }
.lc-pill.ok { background:#DDEFE3; color:#1F6A41; }
.lc-pill.bad { background:#F6DED6; color:#8E3420; }
.lc-pill.unsure { background:#F8E9C8; color:#7A5410; }
.lc-conf { display:flex; justify-content:space-between; font-size:.92rem; color:var(--muted); margin-top:1rem; }
.lc-meter { height:10px; background:#E6EDE0; border-radius:999px; overflow:hidden; margin:.35rem 0 0; }
.lc-meter > span { display:block; height:100%; background:var(--leaf); border-radius:999px; }
.lc-advice { background:var(--mint); border-radius:10px; padding:1rem 1.2rem; margin-top:1rem; line-height:1.55; }
.lc-advice b { font-family:'Bricolage Grotesque', sans-serif; font-size:1.05rem; }
.lc-alt-title { margin:1.4rem 0 .3rem; font-family:'Bricolage Grotesque', sans-serif; font-weight:600; font-size:1.05rem; }
.lc-alt { display:flex; align-items:center; gap:.9rem; margin:.45rem 0; font-size:.95rem; }
.lc-alt-name { flex:0 0 48%; }
.lc-alt .lc-meter { flex:1; margin:0; height:8px; }
.lc-alt-pct { flex:0 0 3.6rem; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
.lc-tips { background:#fff; border:1px dashed #B9C8B1; border-radius:14px; padding:1.3rem 1.5rem; line-height:1.6; }
.lc-tips h4 { font-family:'Bricolage Grotesque', sans-serif; margin:0 0 .5rem; padding:0; font-size:1.15rem; }
.lc-tips ul { margin:0; padding-left:1.2rem; }
.lc-note { font-size:.85rem; color:var(--muted); margin-top:1rem; }
h2, h3, h4 { font-family:'Bricolage Grotesque', sans-serif; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- advice
GENERIC_ADVICE = (
    "Remove the affected leaves, keep the leaves dry when watering, give the plant good airflow and watch it for "
    "the next few days. If it spreads, ask a local agriculture office or nursery which treatment suits your crop."
)
ADVICE = [
    ("healthy", "No signs of disease on this leaf. Keep watering at the base of the plant, give it good airflow and check it again in a week."),
    ("late blight", "Late blight spreads fast in cool, wet weather. Remove and bag affected leaves (do not compost them), avoid wetting the foliage and ask a local expert about a suitable fungicide."),
    ("early blight", "Remove the lowest infected leaves, mulch the soil so spores do not splash up, water at the base and rotate crops next season."),
    ("northern leaf blight", "Rotate crops, bury or remove infected crop residue after harvest and choose resistant varieties next season."),
    ("bacterial spot", "Avoid working among wet plants, remove infected leaves, use clean seed and tools. Copper-based sprays are sometimes used - check local advice."),
    ("leaf mold", "Lower the humidity, open up the canopy for airflow and remove infected leaves."),
    ("septoria", "Remove infected leaves, mulch the soil, water at the base and rotate crops."),
    ("spider mite", "Rinse the undersides of the leaves, then use insecticidal soap or neem oil. Mites thrive in hot, dry conditions, so keep plants well watered."),
    ("target spot", "Remove infected debris, improve airflow and avoid overhead watering."),
    ("mosaic virus", "There is no cure. Remove and destroy the infected plant, wash hands and tools, and control aphids and weeds nearby."),
    ("yellow leaf curl", "This virus is spread by whiteflies. Remove infected plants, control whiteflies and use resistant varieties."),
    ("cedar apple rust", "Remove nearby juniper or cedar galls if possible, rake fallen leaves and ask about a preventive fungicide in spring."),
    ("rust", "Remove infected leaves, avoid wetting the foliage and use resistant varieties. A fungicide can help if it is severe."),
    ("scab", "Rake up and destroy fallen leaves, prune for airflow and choose resistant varieties."),
    ("black rot", "Prune out infected parts, remove mummified fruit and keep the canopy open so leaves dry quickly."),
    ("powdery mildew", "Improve airflow, avoid crowding and water at the base. Sulfur or potassium bicarbonate sprays are commonly used."),
    ("leaf scorch", "Water consistently, remove scorched leaves and avoid overhead watering."),
    ("citrus greening", "There is no cure. It is spread by psyllid insects - report it to your local agriculture office."),
    ("haunglongbing", "There is no cure. It is spread by psyllid insects - report it to your local agriculture office."),
    ("gray leaf spot", "Rotate crops, manage crop residue and choose resistant hybrids."),
    ("cercospora", "Rotate crops, manage crop residue and choose resistant hybrids."),
    ("esca", "There is no cure. Prune out infected wood and keep tools clean."),
]


def advice_for(condition):
    c = condition.lower()
    for key, text in ADVICE:
        if key in c:
            return text
    return GENERIC_ADVICE


# ------------------------------------------------------------------------ helpers
def parse_label(raw):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    if "___" in raw:
        plant, cond = raw.split("___", 1)
    elif "__" in raw:
        plant, cond = raw.split("__", 1)
    elif "_" in raw:
        plant, cond = raw.split("_", 1)
    else:
        plant, cond = raw, ""

    def clean(t):
        return " ".join(t.replace("_", " ").split()).strip()

    plant, cond = clean(plant), clean(cond)
    if cond:
        cond = cond[0].upper() + cond[1:]
    return plant, cond or "Unknown"


def is_healthy(condition):
    return "healthy" in condition.lower()


@st.cache_resource(show_spinner="Loading the model...")
def load_assets():
    try:
        import tensorflow as tf
    except ImportError:
        return None, None, "no_tensorflow"
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None, "missing"
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        return model, classes, None
    except Exception as exc:  # show the reason on the page instead of crashing
        return None, None, str(exc)


@st.cache_data
def load_history():
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return None


def list_samples():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def prepare_image(img):
    return ImageOps.exif_transpose(img).convert("RGB")


def for_display(img, width=560):
    w, h = img.size
    return img.resize((width, max(1, int(h * width / w))))


def predict(model, image):
    arr = np.asarray(image.resize(IMG_SIZE), dtype="float32")[None, ...]
    return model.predict(arr, verbose=0)[0]


# ----------------------------------------------------------------------- rendering
def render_tips():
    st.markdown(
        """<div class="lc-tips"><h4>Add a leaf photo to begin</h4>
<ul>
<li>Photograph one leaf and let it fill most of the frame.</li>
<li>Use daylight, without flash.</li>
<li>A plain background works best.</li>
<li>Keep the leaf sharp - tap the screen to focus.</li>
</ul></div>""",
        unsafe_allow_html=True,
    )


def render_setup_needed(error):
    if error == "no_tensorflow":
        v = sys.version_info
        st.error(f"TensorFlow is not installed. This app is running on Python {v.major}.{v.minor}, which TensorFlow does not support yet.")
        st.markdown(
            """
**Fix (2 minutes):** redeploy the app on Python 3.11.

1. Open share.streamlit.io and delete this app (your GitHub code stays safe).
2. Click **Create app**, choose this repo, branch `main`, file `app.py`.
3. Click **Advanced settings**, set **Python version** to **3.11**, save, then **Deploy**.
"""
        )
        return
    st.error("The trained model is not part of this app yet." if error == "missing" else f"The model could not be loaded: {error}")
    if error == "missing":
        st.markdown(
            """
**Train the model once, then upload the result:**

1. Download a leaf-photo dataset (one folder per class).
2. Run `python train.py --data_dir "path/to/train"` - it creates the `models/` folder.
3. Push the `models/` folder to GitHub and reboot the app.
"""
        )
    else:
        st.markdown(
            "Check that `requirements.txt` installs TensorFlow, that the app runs on Python 3.11 and that the "
            "TensorFlow version matches the one used for training."
        )


def render_result(probs, class_names, known_label=None):
    order = np.argsort(probs)[::-1][:3]
    top = int(order[0])
    conf = float(probs[top])
    plant, cond = parse_label(class_names[top])
    healthy = is_healthy(cond)
    unsure = conf < LOW_CONFIDENCE
    state = "unsure" if unsure else ("ok" if healthy else "bad")
    pill = {"ok": "Looks healthy", "bad": "Disease detected", "unsure": "Not sure"}[state]
    headline = ("Best guess: " if unsure else "") + cond

    if unsure:
        advice = (
            "The model is not confident about this photo. Try again with one leaf filling most of the frame, "
            "in daylight, on a plain background."
        )
    else:
        advice = advice_for(cond)

    st.markdown(
        f"""<div class="lc-slip {state}">
<div class="lc-plant">{html.escape(plant)}</div>
<div class="lc-condition">{html.escape(headline)}</div>
<span class="lc-pill {state}">{pill}</span>
<div class="lc-conf"><span>Confidence</span><span>{conf * 100:.1f}%</span></div>
<div class="lc-meter"><span style="width:{conf * 100:.1f}%"></span></div>
<div class="lc-advice"><b>What to do next</b><br>{html.escape(advice)}</div>
</div>""",
        unsafe_allow_html=True,
    )

    rows = ""
    for idx in order[1:]:
        p_plant, p_cond = parse_label(class_names[int(idx)])
        pct = float(probs[int(idx)]) * 100
        rows += (
            f'<div class="lc-alt"><div class="lc-alt-name">{html.escape(p_cond)} ({html.escape(p_plant)})</div>'
            f'<div class="lc-meter"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="lc-alt-pct">{pct:.1f}%</div></div>'
        )
    st.markdown(f'<div class="lc-alt-title">Other possibilities</div>{rows}', unsafe_allow_html=True)

    if known_label:
        k_plant, k_cond = parse_label(known_label)
        st.caption(f"This sample is labelled in the dataset as: {k_cond} ({k_plant})")
    st.markdown(
        '<div class="lc-note">LeafCheck is a learning project trained on lab-style leaf photos. '
        "It is not a replacement for an agronomist.</div>",
        unsafe_allow_html=True,
    )


def get_input_image():
    mode = st.radio(
        "How do you want to add a leaf?",
        ["Upload a photo", "Use camera", "Try a sample leaf"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "Upload a photo":
        f = st.file_uploader("Drop a leaf photo here", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        return (prepare_image(Image.open(f)), None) if f else (None, None)

    if mode == "Use camera":
        f = st.camera_input("Take a photo of one leaf", label_visibility="collapsed")
        return (prepare_image(Image.open(f)), None) if f else (None, None)

    files = list_samples()
    if not files:
        st.info("No sample photos yet. Run make_samples.py to add some to the sample_images folder.")
        return None, None
    cols = st.columns(3)
    for i, f in enumerate(files[:12]):
        with cols[i % 3]:
            st.image(str(f))
            if st.button(f"Sample {i + 1}", key=f"sample_{i}"):
                st.session_state["sample"] = f.name
    chosen = st.session_state.get("sample")
    path = SAMPLES_DIR / chosen if chosen else None
    if path and path.exists():
        st.caption(f"Selected: sample {[p.name for p in files].index(chosen) + 1}")
        known = re.sub(r"_\d+$", "", path.stem)
        return prepare_image(Image.open(path)), known
    return None, None


def diagnose_view(model, class_names):
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("### Add a leaf photo")
        image, known = get_input_image()
        if image is not None:
            st.image(for_display(image))
    with right:
        st.markdown("### Diagnosis")
        if image is None:
            render_tips()
        else:
            with st.spinner("Checking the leaf..."):
                probs = predict(model, image)
            render_result(probs, class_names, known)


def performance_view(class_names):
    hist = load_history()
    if not hist:
        st.info("No training history found. It appears here after you run train.py.")
        return
    meta = hist.get("meta", {})
    c1, c2, c3, c4 = st.columns(4)
    if "test_accuracy" in meta:
        c1.metric("Test accuracy", f"{meta['test_accuracy'] * 100:.1f}%")
    if "val_accuracy" in hist and hist["val_accuracy"]:
        c2.metric("Validation accuracy", f"{hist['val_accuracy'][-1] * 100:.1f}%")
    if "num_classes" in meta:
        c3.metric("Classes", meta["num_classes"])
    if "train_images" in meta:
        c4.metric("Training photos", f"{meta['train_images']:,}")

    st.markdown("#### Accuracy while training")
    df = pd.DataFrame({"Training": hist.get("accuracy", []), "Validation": hist.get("val_accuracy", [])})
    df.index = range(1, len(df) + 1)
    df.index.name = "Epoch"
    st.line_chart(df)
    st.caption(
        "Test accuracy is measured on photos the model never saw while learning, "
        f"{meta.get('test_images', 0):,} photos in this run."
    )


def about_view(class_names):
    st.markdown(
        """
### How it works
LeafCheck uses **MobileNetV2**, a compact neural network first trained on millions of everyday photos, and
teaches it to recognise leaf diseases (transfer learning). When you add a photo, it is resized to 224 x 224
pixels and the model returns a probability for every plant and condition it knows.

### Good to know
- It was trained on leaf photos taken against plain backgrounds, so photos of whole plants or busy fields
  can confuse it.
- It only knows the plants and conditions listed below.
- Treat the result as a first hint, not a final diagnosis.
"""
    )
    if class_names:
        rows = []
        for c in class_names:
            plant, cond = parse_label(c)
            rows.append({"Plant": plant, "Condition": cond, "Status": "Healthy" if is_healthy(cond) else "Disease"})
        st.markdown("### What it can recognise")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------------------- main
model, class_names, load_error = load_assets()

n_classes = f"{len(class_names)} known conditions" if class_names else "known plant diseases"
st.markdown(
    f"""<div class="lc-hero"><h1>Is your plant sick?<br>Show us the leaf.</h1>
<p>Add a photo of a single leaf. LeafCheck names the plant, checks it against {n_classes} and tells you what to do next.</p></div>""",
    unsafe_allow_html=True,
)

tab_diag, tab_perf, tab_about = st.tabs(["Diagnose a leaf", "How well it works", "About"])
with tab_diag:
    if model is None:
        render_setup_needed(load_error)
    else:
        diagnose_view(model, class_names)
with tab_perf:
    performance_view(class_names)
with tab_about:
    about_view(class_names)
