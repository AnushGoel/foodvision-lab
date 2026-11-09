"""Try the model: upload a photo, get calibrated probabilities and a Grad-CAM explanation."""

import streamlit as st

from foodvision import ui
from foodvision.artifacts import Files
from foodvision.imageio import InvalidImageError, load_image
from foodvision.viz import probability_bars

ui.page("Try the model", "Upload a photo of pizza, steak or sushi. The model returns calibrated probabilities "
        "and a heat map of the regions that raised its score.")
ui.require(Files.MODEL, Files.MODEL_META)
meta = ui.read_json(Files.MODEL_META)
ood = meta.get("ood")
if ood:
    st.info(
        f"This model knows only three classes: {', '.join(meta['classes'])}. Photos that look unlike its training "
        "images are flagged instead of labelled. That check was tested on flowers and pets, so other foods, such as a "
        "burger, can still slip through and receive one of the three labels."
    )
else:
    st.warning(
        f"This model knows only three classes: {', '.join(meta['classes'])}. Any other image, food or not, is still "
        "forced into one of them, often with high confidence. It was trained on a few hundred Food-101 photos."
    )


@st.cache_resource(show_spinner="Loading the model. The first run downloads the ImageNet weights (about 50 MB).")
def predictor(path: str, mtime: float):  # noqa: ANN201 - torch is imported lazily
    from foodvision.inference import Predictor

    return Predictor(path)


threshold = st.slider("Answer only above this confidence", 0.34, 0.99, 0.60, 0.01,
                      help="Below the threshold the model reports that it is not sure instead of guessing.")
check_ood = bool(ood) and st.checkbox(
    "Flag photos that aren't pizza, steak or sushi", value=True,
    help="Switch off to see what the model does without the check: it always picks one of its three classes.",
)
upload = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
if upload is None:
    st.stop()
try:
    image = load_image(upload.getvalue())
except InvalidImageError as error:
    st.error(str(error))
    st.stop()

path = ui.store().path(Files.MODEL)
try:
    model = predictor(str(path), path.stat().st_mtime)
except ImportError:
    st.error("PyTorch is not installed in this environment. Install the app requirements: `pip install -r requirements.txt`.")
    st.stop()
except Exception as error:  # e.g. the ImageNet weights could not be downloaded
    st.error(f"The model could not be loaded: {error}")
    st.stop()

with st.spinner("Classifying and computing Grad-CAM"):
    prediction = model.predict(image, abstain_below=threshold, check_ood=check_ood)
    view, heat = model.explain(image)

left, right = st.columns([1, 1], gap="large")
with left:
    if prediction.in_distribution is False:
        st.markdown(f"### Not pizza, steak or sushi?\nThis photo looks unlike the training photos, so the model "
                    f"won't label it. Its closest class would be **{prediction.label}** ({prediction.confidence:.0%}).")
    elif prediction.abstained:
        st.markdown(f"### Not sure\nThe top class, **{prediction.label}**, has {prediction.confidence:.0%} confidence, "
                    f"below your {threshold:.0%} threshold.")
    else:
        st.markdown(f"### {prediction.label.capitalize()}\n{prediction.confidence:.0%} confidence")
    st.plotly_chart(probability_bars(prediction.classes, prediction.probabilities))
    check = ""
    if prediction.ood_score is not None:
        check = (f" Not-food check: {model.ood_label.lower()} {prediction.ood_score:.2f}, threshold "
                 f"{ood['threshold']:.2f} (chosen on validation data).")
    ui.note(f"{meta['label']}, {meta['epochs']} epochs, temperature {meta['temperature']:.2f}. "
            f"Inference took {prediction.latency_ms:.0f} ms on this server's CPU.{check}")
with right:
    a, b = st.columns(2)
    a.image(view, caption="What the model sees (resized, centre-cropped)")
    b.image(heat, caption=f"Grad-CAM for {prediction.label}")
    ui.note("The heat map shows which regions of the final feature map increased the predicted class score. It is "
            "a correlational summary, not proof of why the model decided.")
