import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from foodvision.data import Sample, make_loader  # noqa: E402
from foodvision.engine import evaluate, fit  # noqa: E402
from foodvision.experiments import features_for_files, logits_for_files, robustness_eval  # noqa: E402
from foodvision.explain import GradCAM, overlay  # noqa: E402
from foodvision.inference import Predictor  # noqa: E402
from foodvision.models import build_model, save_checkpoint  # noqa: E402

CLASSES = ["pizza", "steak", "sushi"]


class TinyNet(nn.Module):
    """Same attribute layout as torchvision EfficientNet (features / avgpool / classifier)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(8, 3))

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


def _separable(n=60):
    generator = torch.Generator().manual_seed(0)
    y = torch.arange(n) % 3
    x = torch.randn(n, 3, 8, 8, generator=generator) * 0.1
    x[torch.arange(n), y] += 2.0  # each class lights up its own colour channel
    return x, y


def test_fit_records_history_and_snapshots_and_learns(tmp_path):
    torch.manual_seed(0)
    x, y = _separable()
    loaders = {"train": make_loader(x, y, 16, True, 0), "val": make_loader(x, y, 16, False)}
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tmp_path / "tb"))
    except ImportError:
        pass
    history, snaps = fit(TinyNet(), loaders, 6, 0.05, torch.device("cpu"), ("val",), writer, (3, 6))
    assert len(history["train"]["loss"]) == len(history["val"]["acc"]) == len(history["seconds"]) == 6
    assert set(snaps) == {3, 6}
    assert history["val"]["acc"][-1] > 0.9
    if writer is not None:
        assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_evaluate_returns_logits_in_loader_order():
    x, y = _separable(30)
    result = evaluate(TinyNet(), make_loader(x, y, 8, False), torch.device("cpu"))
    assert result["logits"].shape == (30, 3)
    assert np.array_equal(result["labels"], y.numpy())


def test_gradcam_on_frozen_efficientnet():
    torch.manual_seed(0)
    model = build_model("effnetb2", 3, pretrained=False).eval()
    cam, probs = GradCAM(model)(torch.randn(1, 3, 288, 288))
    assert cam.shape == (9, 9) and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert not model.training


def test_overlay_keeps_image_size():
    out = overlay(Image.new("RGB", (100, 80), "white"), np.random.default_rng(0).random((5, 4)))
    assert out.size == (100, 80) and out.mode == "RGB"


def test_predictor_end_to_end(tmp_path):
    torch.manual_seed(0)
    save_checkpoint(build_model("effnetb2", 3, pretrained=False), tmp_path / "m.pt", "effnetb2", CLASSES,
                    {"temperature": 1.5})
    predictor = Predictor(tmp_path / "m.pt", pretrained=False)
    image = Image.new("RGB", (400, 300), (200, 80, 40))
    prediction = predictor.predict(image)
    assert prediction.classes == CLASSES and prediction.label in CLASSES
    assert sum(prediction.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert predictor.predict(image, abstain_below=1.01).abstained
    view, heat = predictor.explain(image)
    assert view.size == heat.size == (288, 288)


def test_predictor_applies_the_stored_not_food_check(tmp_path):
    torch.manual_seed(0)
    model = build_model("effnetb2", 3, pretrained=False)
    image = Image.new("RGB", (320, 240), (90, 160, 60))
    for threshold, expected in ((-1e9, True), (1e9, False)):
        meta = {"temperature": np.float64(1.2), "ood": {"score": "energy", "threshold": threshold,
                                                       "temperature": 1.2, "target_tpr": 0.95}}
        save_checkpoint(model, tmp_path / "m.pt", "effnetb2", CLASSES, meta)  # NumPy number must be cleaned
        predictor = Predictor(tmp_path / "m.pt", pretrained=False)
        prediction = predictor.predict(image)
        assert prediction.in_distribution is expected and isinstance(prediction.ood_score, float)
        assert predictor.predict(image, check_ood=False).in_distribution is None
    save_checkpoint(model, tmp_path / "old.pt", "effnetb2", CLASSES, {"temperature": 1.0})
    assert Predictor(tmp_path / "old.pt", pretrained=False).predict(image).in_distribution is None


def test_logits_for_files_skips_unreadable_images(tmp_path):
    from torchvision import transforms

    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (40, 30), (40 * i, 80, 120)).save(path)
        paths.append(path)
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    paths.insert(2, tmp_path / "broken.jpg")
    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    logits, kept = logits_for_files(TinyNet().eval(), paths, tf, torch.device("cpu"), batch_size=2)
    assert kept == [0, 1, 3, 4, 5] and logits.shape == (5, 3)


def test_features_for_files_returns_pooled_features_and_matching_logits(tmp_path):
    from torchvision import transforms

    paths = []
    for i in range(3):
        paths.append(tmp_path / f"{i}.jpg")
        Image.new("RGB", (40, 30), (60 * i, 90, 30)).save(paths[-1])
    net = TinyNet().eval()
    tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    feats, logits, kept = features_for_files(net, paths, tf, torch.device("cpu"), batch_size=2)
    assert feats.shape == (3, 8) and logits.shape == (3, 3) and kept == [0, 1, 2]
    expected, _ = logits_for_files(net, paths, tf, torch.device("cpu"))
    np.testing.assert_allclose(logits, expected, rtol=1e-5, atol=1e-6)


def test_robustness_eval_reports_every_condition(tmp_path):
    torch.manual_seed(0)
    samples = []
    for i, colour in enumerate([(200, 60, 40), (90, 60, 40), (40, 120, 90)]):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (320, 260), colour).save(path)
        samples.append(Sample(path, i))
    model = build_model("effnetb2", 3, pretrained=False).eval()
    meta = {"score": "energy", "threshold": -1e9}
    frame = robustness_eval(model, "effnetb2", samples, 1.0, meta, torch.device("cpu"), seed=0, batch_size=2)
    assert len(frame) == 16 and frame.iloc[0]["corruption"] == "none"
    assert frame["accuracy"].between(0, 1).all() and (frame["accepted_as_food"] == 1.0).all()
