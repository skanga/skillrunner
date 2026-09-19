from types import SimpleNamespace


def _coordinator(*, redact_env: list[str]):
    from skillrunner.runtime.coordinator import Coordinator

    coordinator = Coordinator.__new__(Coordinator)
    coordinator.settings = SimpleNamespace(
        diagnostics=SimpleNamespace(redact_env=redact_env),
        models={
            "model": SimpleNamespace(api_key_env="MODEL_KEY"),
        },
        direct_model=SimpleNamespace(api_key_env=None),
        policy=SimpleNamespace(
            command_env={"/bin/tool": {"PATH": "RUNTIME_PATH", "FLAG": "RUNTIME_FLAG"}}
        ),
        mcp={
            "remote": SimpleNamespace(
                env={"WORKSPACE": "MCP_WORKSPACE"},
                headers={
                    "Authorization": "MCP_AUTH",
                    "X-Api-Key": "MCP_API_KEY",
                    "X-Custom-Token": "MCP_CUSTOM_TOKEN",
                },
            )
        },
    )
    coordinator.environ = {
        "MODEL_KEY": "model-secret",
        "RUNTIME_PATH": "/opt/runtime",
        "RUNTIME_FLAG": "true",
        "MCP_WORKSPACE": "/tmp/workspace",
        "MCP_AUTH": "Bearer mcp-secret",
        "MCP_API_KEY": "mcp-api-secret",
        "MCP_CUSTOM_TOKEN": "mcp-custom-secret",
    }
    return coordinator


def test_ordinary_mapped_environment_values_require_explicit_redaction():
    coordinator = _coordinator(redact_env=[])

    assert set(coordinator._secrets()) == {
        "model-secret",
        "Bearer mcp-secret",
        "mcp-api-secret",
        "mcp-custom-secret",
    }

    coordinator = _coordinator(redact_env=["RUNTIME_PATH", "RUNTIME_FLAG"])
    assert "/opt/runtime" in coordinator._secrets()
    assert "true" in coordinator._secrets()
    assert "/tmp/workspace" not in coordinator._secrets()
