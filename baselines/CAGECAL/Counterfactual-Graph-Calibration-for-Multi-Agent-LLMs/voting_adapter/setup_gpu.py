"""GPU-container environment setup; every write stays in this baseline.

Run: python -B voting_adapter/setup_gpu.py
The existing CUDA-enabled torch is inherited, never installed or upgraded.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv-voting"


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def main():
    for key, sub in (("PIP_CACHE_DIR", "pip"), ("TMPDIR", "tmp"), ("HF_HOME", "huggingface"),
                     ("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                     ("CUDA_CACHE_PATH", "cuda"), ("TRITON_CACHE_DIR", "triton"),
                     ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
        target = ROOT / ".cache-voting" / sub
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if Path(sys.prefix).resolve() != ENV:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("Run this setup inside your CUDA-enabled GPU container, not on the Mac.")
        if not ENV.exists():
            run(sys.executable, "-B", "-m", "venv", "--system-site-packages", str(ENV))
        run(str(ENV / "bin/python"), "-B", str(Path(__file__).resolve()))
        return
    import torch
    original_torch = (torch.__version__, torch.__file__)
    if not torch.cuda.is_available():
        raise SystemExit("Inherited torch cannot access CUDA; leaving it unchanged.")
    run(sys.executable, "-B", "-m", "pip", "install", "--no-deps", "packaging>=23")
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    roots = [Requirement(line.strip()) for line in (ROOT / "voting_adapter/requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]

    def protected(name):
        return name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith(("nvidia-", "cuda-"))

    run(sys.executable, "-B", "-m", "pip", "install", "--no-deps", *[str(r) for r in roots])
    # Inspect dependency metadata, but never ask pip to resolve Torch/CUDA.
    # Ignore outgoing requirements of an incompatible installed version: those
    # describe the version being replaced, not the one this environment needs.
    for attempt in range(12):
        omit = set()
        while True:
            pending = list(roots)
            seen, constraints = set(), {}
            while pending:
                req = pending.pop()
                if req.marker and not req.marker.evaluate({"extra": ""}):
                    continue
                name = canonicalize_name(req.name)
                constraints.setdefault(name, set()).add(str(req))
                if name in seen or name in omit or protected(name):
                    continue
                seen.add(name)
                try:
                    dist = md.distribution(name)
                except md.PackageNotFoundError:
                    continue
                pending.extend(Requirement(r) for r in (dist.requires or []))
            invalid = {}
            for name, requirements in constraints.items():
                try:
                    version = md.version(name)
                except md.PackageNotFoundError:
                    version = None
                satisfied = version is not None and all(Requirement(r).specifier.contains(version, prereleases=True)
                                                        for r in requirements)
                if not satisfied:
                    if protected(name):
                        raise SystemExit(f"{name} {version} fails {sorted(requirements)}; refusing to modify CUDA packages")
                    invalid[name] = requirements
            if set(invalid) <= omit:
                break
            omit.update(invalid)
        if not invalid:
            break
        install = [req for name in sorted(invalid) for req in sorted(invalid[name])]
        run(sys.executable, "-B", "-m", "pip", "install", "--no-deps", *install)
    else:
        raise SystemExit("Dependency resolution did not converge; the base environment was not modified")
    run(sys.executable, "-B", "-c",
        "import torch, transformers, sentence_transformers, torch_geometric, betacal, calibration, datasets; "
        "print('Imports OK; torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())")
    if (torch.__version__, torch.__file__) != original_torch:
        raise RuntimeError("Unexpected torch change")
    packages = {name: md.version(name) for name in sorted(constraints) if not protected(name)}
    (ENV / "voting-packages.json").write_text(json.dumps(dict(torch=original_torch, packages=packages), indent=2))
    print(f"Ready. Activate with: source {ENV}/bin/activate")
    print("Next: python -B -m voting_adapter.run --smoke --device cuda:0")


if __name__ == "__main__":
    main()
