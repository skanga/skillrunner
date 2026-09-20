"""Qualify source and wheel uv-tool installations outside the checkout."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


def execute(argv: list[str], *, cwd: Path, env: Mapping[str, str]) -> str:
    completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise RuntimeError(
            f"{' '.join(map(str, argv[:3]))} failed ({completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    return completed.stdout


def main() -> None:
    root = Path.cwd().resolve()
    assert (root / "pyproject.toml").is_file()
    wheels = list((root / "dist").glob("skillrunner-*.whl"))
    assert len(wheels) == 1, "Build exactly one wheel before this probe"
    uv = shutil.which("uv")
    assert uv is not None
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="never")
    with tempfile.TemporaryDirectory(prefix="skillrunner-install-probe-") as name:
        temporary = Path(name)
        constraints = temporary / "constraints.txt"
        execute(
            [
                uv,
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--format",
                "requirements-txt",
                "--output-file",
                str(constraints),
            ],
            cwd=root,
            env=env,
        )
        outside = temporary / "outside"
        package = outside / "skills" / "hello"
        package.mkdir(parents=True)
        (package / "SKILL.md").write_text(
            "---\nname: hello\ndescription: Format a greeting.\n---\nWrite a greeting.\n"
        )
        rows = []
        for mode, target in (("source", root), ("wheel", wheels[0])):
            tools = temporary / mode / "tools"
            binary = temporary / mode / "bin"
            install_env = dict(env, UV_TOOL_DIR=str(tools), UV_TOOL_BIN_DIR=str(binary))
            execute(
                [
                    uv,
                    "tool",
                    "install",
                    "--python",
                    sys.executable,
                    "--constraints",
                    str(constraints),
                    str(target),
                ],
                cwd=outside,
                env=install_env,
            )
            command = binary / ("skillrun.exe" if os.name == "nt" else "skillrun")
            assert command.is_file(), command
            help_text = execute([str(command), "run", "--help"], cwd=outside, env=install_env)
            assert "--prompt-file" in help_text and "--shutdown-grace" in help_text
            catalog = json.loads(
                execute([str(command), "skills", "list", "-j"], cwd=outside, env=install_env)
            )
            assert [item["name"] for item in catalog["skills"]] == ["hello"]
            tool_python = (
                tools / "skillrunner" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            imported = Path(
                execute(
                    [str(tool_python), "-c", "import skillrunner; print(skillrunner.__file__)"],
                    cwd=outside,
                    env=install_env,
                ).strip()
            ).resolve()
            assert imported.is_relative_to(tools.resolve()), imported
            assert not (outside / "outputs").exists()
            rows.append({"mode": mode, "help": True, "discovery": True, "isolated_import": True})
        print(
            json.dumps({"python": sys.version.split()[0], "platform": sys.platform, "modes": rows}),
            flush=True,
        )


if __name__ == "__main__":
    main()
