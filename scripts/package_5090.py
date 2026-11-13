"""Build a portable source/history/assets/results handoff without host credentials or venv."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from g1lab.fast_plan import LAB_REVISION, MUJOCO_REVISION


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def digest(path):
    with path.open("rb") as handle:
        return (
            hashlib.file_digest(handle, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(handle.read()).hexdigest()
        )


def main(output):
    root = Path(__file__).resolve().parents[1]
    if git("status", "--porcelain", cwd=root):
        raise RuntimeError("Commit project changes before packaging")
    if git("branch", "--show-current", cwd=root) != "main":
        raise RuntimeError("Package the consolidated main branch")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "g1-5090-handoff.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    revision = git("rev-parse", "HEAD", cwd=root)
    with tempfile.TemporaryDirectory(prefix="g1-handoff-", dir=output) as temporary:
        stage = Path(temporary) / "g1-robust-locomotion-lab"
        stage.mkdir()
        # Extract Git's trusted, tracked paths; no working-tree caches or local credentials.
        source_tar = Path(temporary) / "source.tar"
        subprocess.run(["git", "archive", "--output", str(source_tar), "HEAD"], cwd=root, check=True)
        with tarfile.open(source_tar) as source:
            source.extractall(stage, filter="data")
        transfer = stage / "artifacts/transfer"
        transfer.mkdir(parents=True)
        subprocess.run(
            ["git", "bundle", "create", str(transfer / "project.bundle"), "main"], cwd=root, check=True
        )
        versions = {}
        for name, expected in [("unitree_rl_lab", LAB_REVISION), ("unitree_mujoco", MUJOCO_REVISION)]:
            source = root / "vendor" / name
            actual = git("rev-parse", "HEAD", cwd=source)
            if actual != expected or git("status", "--porcelain", cwd=source):
                raise RuntimeError(f"Vendor revision/working tree differs: {name}")
            target = stage / "vendor" / name
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            metadata = target / ".git"
            metadata.mkdir()
            # Keep only Git data needed to preserve exact public upstream identity.
            for entry in ["HEAD", "index", "packed-refs", "shallow", "objects", "refs"]:
                item = source / ".git" / entry
                if item.is_dir():
                    shutil.copytree(item, metadata / entry)
                elif item.exists():
                    shutil.copy2(item, metadata / entry)
            config = "[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfilemode = true\n"
            config += f'[remote "origin"]\n\turl = https://github.com/unitreerobotics/{name}.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
            sparse = source / ".git/info/sparse-checkout"
            if sparse.exists():
                (metadata / "info").mkdir()
                shutil.copy2(sparse, metadata / "info/sparse-checkout")
                config += "[core]\n\tsparseCheckout = true\n\tsparseCheckoutCone = true\n"
                config += '[remote "origin"]\n\tpromisor = true\n\tpartialclonefilter = blob:none\n'
            (metadata / "config").write_text(config)
            licenses = stage / "licenses"
            licenses.mkdir(exist_ok=True)
            license_name = "LICENCE" if name == "unitree_rl_lab" else "LICENSE"
            (licenses / f"{name}-LICENSE.txt").write_text(
                git("show", f"HEAD:{license_name}", cwd=source) + "\n"
            )
            versions[name] = actual
        included = [
            "artifacts/controller-1660-v1",
            "artifacts/controller-1660-render",
            "artifacts/unitree-lab-29dof-heading-v2",
            "runs/autonomous-1660/selected",
            "runs/autonomous-1660/selected-terrain",
        ]
        for relative in included:
            shutil.copytree(root / relative, stage / relative)
        metadata = {
            "project_commit": revision,
            "branch": "main",
            "vendors": versions,
            "included_local_results": included,
            "not_included": [
                ".venv",
                "Isaac runtime",
                "G1 training USD",
                "CUDA PyTorch",
                "tokens/SSH keys",
                "all unsuccessful training runs",
            ],
            "start_here": "docs/CLAUDE_5090_START_HERE.md",
        }
        (stage / "TRANSFER_MANIFEST.json").write_text(json.dumps(metadata, indent=2))
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        (stage / "TRANSFER_SHA256SUMS").write_text(
            "".join(f"{digest(p)}  {p.relative_to(stage)}\n" for p in files)
        )
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(stage, arcname=stage.name)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest(archive)}  {archive.name}\n")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": digest(archive),
                "project_commit": revision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/transfer-5090")
    main(parser.parse_args().output)
