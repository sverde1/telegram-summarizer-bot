"""OCR_DEVICE: RapidOCR on an NVIDIA GPU when available, the CPU otherwise; the sandbox's GPU access."""
import sys
import types

import pytest

from summarizer import config, ocr, proc, rapid_ocr, sandbox


@pytest.fixture
def rapid_cuda(monkeypatch):
    """RapidOCR asked to run on the GPU; returns a setter for which providers onnxruntime reports."""
    monkeypatch.setattr(config, "OCR_ENGINE", "rapidocr")
    monkeypatch.setattr(config, "OCR_DEVICE", "cuda")
    monkeypatch.setattr(ocr, "_cuda_ok", None)

    def providers(*names):
        """Installs a fake onnxruntime reporting these execution providers."""
        fake = types.SimpleNamespace(get_available_providers=lambda: list(names))
        monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    return providers


def test_unknown_engine_means_rapidocr(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "magic")
    assert ocr.engine() == "rapidocr"


def test_cuda_when_onnxruntime_has_it(rapid_cuda):
    rapid_cuda("CUDAExecutionProvider", "CPUExecutionProvider")
    assert ocr.device() == "cuda" and ocr.seconds_per_page() == ocr.SPEED_DEFAULT["rapidocr-cuda"]


def test_falls_back_to_the_cpu_without_cuda(rapid_cuda, caplog):
    rapid_cuda("CPUExecutionProvider")
    assert ocr.device() == "cpu" and "install onnxruntime-gpu" in caplog.text


def test_tesseract_always_runs_on_the_cpu(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "tesseract")
    monkeypatch.setattr(config, "OCR_DEVICE", "cuda")
    assert ocr.device() == "cpu"


def test_gpu_sandbox_gets_the_devices_and_no_memory_cap(tmp_path):
    cpu = sandbox.command(tmp_path, ["python", "-c", "1"])
    gpu = sandbox.command(tmp_path, ["python", "-c", "1"], gpu=True)
    assert "prlimit" in cpu and "prlimit" not in gpu  # CUDA's address-space reservations break under a cap
    assert "--unshare-all" in gpu and "--share-net" not in gpu  # still no network
    assert ["--ro-bind-try", "/sys", "/sys"] == gpu[gpu.index("--ro-bind-try", gpu.index("/job")):][:3]


def test_rapidocr_run_asks_for_the_gpu(rapid_cuda, monkeypatch, tmp_path):
    rapid_cuda("CUDAExecutionProvider")
    seen = []

    def run(cmd, **kw):
        """Records the command; writes the answer the entry point would."""
        seen.append(cmd)
        (tmp_path / "rapid-t.json").write_text('[{"text": "hi", "score": 0.9}]')
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(proc, "run", run)
    img = tmp_path / "p.png"
    img.write_bytes(b"x")
    assert ocr._rapidocr([img], "eng", tmp_path, "t") == [("hi", 0.9)]
    cmd = seen[0]
    assert cmd[cmd.index("summarizer.rapid_ocr") + 3] == "cuda" and "prlimit" not in cmd


@pytest.mark.parametrize("device,expected", [
    ("cuda", {"EngineConfig.onnxruntime.use_cuda": True}),
    ("cpu", {"EngineConfig.onnxruntime.intra_op_num_threads": 1}),
])
def test_entry_point_configures_the_device(monkeypatch, device, expected):
    seen = {}

    class FakeRapid:
        """Records its parameters; reads nothing."""

        def __init__(self, params):
            """Keeps the parameters."""
            seen.update(params)

        def __call__(self, image):
            """An empty result."""
            return types.SimpleNamespace(txts=[], scores=[])

    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=FakeRapid))
    rapid_ocr.recognize(["x.png"], None, device)
    assert expected.items() <= seen.items()
