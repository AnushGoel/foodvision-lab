import pytest

torch = pytest.importorskip("torch")

from foodvision.models import (  # noqa: E402
    architecture_summary,
    build_model,
    get_transforms,
    load_checkpoint,
    parameter_counts,
    save_checkpoint,
    set_train_mode,
)

CLASSES = ["pizza", "steak", "sushi"]


@pytest.mark.parametrize(("name", "crop", "features"), [("effnetb2", 288, 1408), ("effnetb3", 300, 1536), ("effnetb4", 380, 1792)])
def test_new_head_frozen_backbone_and_output_shape(name, crop, features):
    model = build_model(name, 3, pretrained=False).eval()
    assert get_transforms(name).crop_size[0] == crop
    assert parameter_counts(model)["trainable"] == features * 3 + 3
    assert not any(p.requires_grad for p in model.features.parameters())
    with torch.no_grad():
        assert model(torch.zeros(2, 3, crop, crop)).shape == (2, 3)


def test_frozen_weights_do_not_freeze_batchnorm_statistics():
    model = build_model("effnetb2", 3, pretrained=False)
    bn = next(m for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    set_train_mode(model, freeze_bn_stats=False)
    model(torch.randn(4, 3, 64, 64))
    assert not torch.equal(before, bn.running_mean), "train mode updates running statistics despite requires_grad=False"
    before = bn.running_mean.clone()
    set_train_mode(model, freeze_bn_stats=True)
    model(torch.randn(4, 3, 64, 64))
    assert torch.equal(before, bn.running_mean)
    assert model.classifier.training and not model.features.training


def test_delta_checkpoint_round_trip_is_exact_and_small(tmp_path):
    torch.manual_seed(0)
    model = build_model("effnetb2", 3, pretrained=False)
    set_train_mode(model)
    model(torch.randn(4, 3, 96, 96))  # change the BatchNorm statistics, as training would
    with torch.no_grad():
        model.classifier[-1].weight.add_(0.5)
    info = save_checkpoint(model, tmp_path / "model.pt", "effnetb2", CLASSES, {"temperature": 1.3})
    assert info["bytes"] < 5_000_000
    torch.manual_seed(0)  # same frozen weights as the original (stand-in for pinned pretrained weights)
    restored, payload = load_checkpoint(tmp_path / "model.pt", pretrained=False)
    x = torch.randn(2, 3, 288, 288)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), restored(x), atol=1e-5)
    assert payload["classes"] == CLASSES and payload["meta"]["temperature"] == 1.3


def test_foreign_checkpoints_are_rejected(tmp_path):
    torch.save({"format": "other"}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "x.pt", pretrained=False)


def test_architecture_summary_is_measured():
    summary = architecture_summary("effnetb2")
    assert summary["mbconv_blocks"] == 23 and summary["feature_dim"] == 1408
    assert summary["stages"][-1]["output"] == [1408, 9, 9]


def test_checkpoint_with_environment_metadata_reloads_safely(tmp_path):
    """Regression: metadata holding torch.__version__ (a TorchVersion) made torch.load(weights_only=True) fail."""
    import numpy as np

    from foodvision.pipeline import environment

    torch.manual_seed(0)
    model = build_model("effnetb2", 3, pretrained=False)
    meta = {"environment": environment(), "raw_version": torch.__version__, "temperature": np.float32(1.2)}
    save_checkpoint(model, tmp_path / "model.pt", "effnetb2", CLASSES, meta)
    _, payload = load_checkpoint(tmp_path / "model.pt", pretrained=False)  # uses weights_only=True
    assert payload["meta"]["raw_version"] == str(torch.__version__)
    assert type(payload["meta"]["environment"]["torch"]) is str
