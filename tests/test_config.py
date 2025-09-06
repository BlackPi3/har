import yaml
import random
import numpy as np
import torch
from src import config

def write_yaml(path, d):
    path.write_text(yaml.safe_dump(d))

def test_merge_and_opts(tmp_path):
    base = {"a": 1, "nested": {"x": 10}}
    exp = {"b": 2, "nested": {"y": 20}}
    bfile = tmp_path / "base.yaml"
    efile = tmp_path / "exp.yaml"
    write_yaml(bfile, base)
    write_yaml(efile, exp)

    opts = ["nested.x=99", "newlist=[1,2,3]", "flag=true", "strval=hello"]
    cfg = config.load_config(str(bfile), str(efile), opts=opts)

    assert cfg.a == 1
    assert cfg.b == 2
    # nested.x overridden by opts
    assert cfg.nested.x == 99
    assert cfg.nested.y == 20
    assert cfg.newlist == [1, 2, 3]
    assert cfg.flag is True
    assert cfg.strval == "hello"

def test_explicit_device_override(tmp_path, monkeypatch):
    # ensure env suggests local but YAML specifies explicit device
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    bfile = tmp_path / "base.yaml"
    write_yaml(bfile, {"device": "cpu"})
    cfg = config.load_config(str(bfile), None, opts=[])
    assert cfg.device == "cpu"
    assert isinstance(cfg.torch_device, torch.device)

def test_device_policy_monkeypatch(monkeypatch):
    # Simulate cluster: SLURM present -> prefer cuda if available
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setattr(torch, "cuda", type("X", (), {"is_available": staticmethod(lambda: True), "device_count": staticmethod(lambda: 1)}))
    cfg = config.load_config(None, None)
    assert cfg.cluster is True
    assert cfg.device in ("cuda", "cpu")

    # Local: simulate mps available
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    class MPS:
        @staticmethod
        def is_available(): return True
    monkeypatch.setattr(torch, "cuda", type("X", (), {"is_available": staticmethod(lambda: False)}))
    monkeypatch.setattr(torch.backends, "mps", MPS, raising=False)
    cfg2 = config.load_config(None, None)
    assert cfg2.cluster is False
    assert cfg2.device == "mps"

def test_set_seed_reproducible(tmp_path):
    # reproducibility: same outputs after setting same seed
    config.set_seed(1234)
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.randn(3)

    config.set_seed(1234)
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.randn(3)

    assert r1 == r2
    assert np.allclose(n1, n2)
    assert torch.allclose(t1, t2)