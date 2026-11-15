"""Install Phase 2 Python dependencies without replacing Colab's CUDA torch stack."""
import importlib
import importlib.metadata
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement


IMPORT_NAMES = {
    "Pillow": "PIL", "pytorch-msssim": "pytorch_msssim",
    "imageio-ffmpeg": "imageio_ffmpeg",
}


def gpu_stack():
    try:
        import torch
        import torchvision
        from torchvision.ops import nms
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the installed PyTorch build/runtime")
        boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
        nms(boxes, torch.tensor([1.], device="cuda"), 0.5)
        torch.empty(1, device="cuda").add_(1).cpu()
        return {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)}
    except Exception as exc:
        raise RuntimeError(
            "Colab PyTorch/torchvision CUDA stack is missing or incompatible. "
            "Choose an A100 GPU runtime and repair the matched torch/torchvision pair "
            "there before running Phase 2; this setup will not downgrade it."
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write validated environment versions as JSON")
    args = parser.parse_args()
    environment = gpu_stack()
    requirements = [Requirement(line.strip()) for line in
                    (Path(__file__).parent / "requirements_modern.txt").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    missing = []
    for requirement in requirements:
        try:
            version = importlib.metadata.version(requirement.name)
            if version not in requirement.specifier:
                missing.append(str(requirement))
        except importlib.metadata.PackageNotFoundError:
            missing.append(str(requirement))
    if missing:
        print("Installing only missing/incompatible dependencies:", missing, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    for requirement in requirements:
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"Dependency still incompatible: {requirement.name}=={version}")
        importlib.import_module(IMPORT_NAMES.get(requirement.name, requirement.name.replace("-", "_")))
        environment[requirement.name] = version
    environment.update(gpu_stack())
    print(json.dumps(environment, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return environment


if __name__ == "__main__":
    main()
