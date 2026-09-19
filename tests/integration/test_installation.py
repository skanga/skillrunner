"""Exercise the installed console entry point and a wheel without network access."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def test_installed_entrypoint_help_outside_repository(tmp_path):
    script = Path(sysconfig.get_path("scripts")) / (
        "skillrun.exe" if os.name == "nt" else "skillrun"
    )
    result = subprocess.run(
        [str(script), "run", "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert "--prompt-file" in result.stdout
    assert "--shutdown-grace" in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_offline_wheel_installation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    uv = shutil.which("uv")
    assert uv is not None, "Run the project test suite through uv."
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "OPENAI_BASE_URL"}
        and not key.startswith("SKILLRUN_")
    }
    environment["UV_OFFLINE"] = "1"
    wheel_dir = tmp_path / "dist"
    build = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir), str(root)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    (wheel,) = wheel_dir.glob("*.whl")
    target = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(target)],
        env=environment,
        check=True,
        capture_output=True,
        timeout=30,
    )
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert install.returncode == 0, install.stderr
    # Install the locked dependencies, including their platform-specific startup hooks.
    requirements = tmp_path / "pylock.toml"
    export = subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--format",
            "pylock.toml",
            "--output-file",
            str(requirements),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert export.returncode == 0, export.stderr
    dependencies = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "-r",
            str(requirements),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert dependencies.returncode == 0, dependencies.stderr
    package_path = subprocess.check_output(
        [str(python), "-c", "import skillrunner; print(skillrunner.__file__)"],
        cwd=tmp_path,
        env=environment,
        text=True,
        timeout=10,
    ).strip()
    assert Path(package_path).is_relative_to(target)
    script = target / ("Scripts/skillrun.exe" if os.name == "nt" else "bin/skillrun")
    result = subprocess.run(
        [str(script), "skills", "list", "-j"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["skills"] == []
    assert not (tmp_path / "outputs").exists()
