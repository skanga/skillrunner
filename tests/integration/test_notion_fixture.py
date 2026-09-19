import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from skillrunner.config.models import MCPConfig, Policy
from skillrunner.runtime.budgets import Deadline
from skillrunner.runtime.environment import build_child_environment
from skillrunner.runtime.processes import ProcessSupervisor

RESEARCH = {
    "kind": "configured-mcp-fixture",
    "path": "notion-research-fixture",
    "records": [
        {"id": "pilot-a", "title": "Pilot A", "users": 10, "satisfied": 8},
        {"id": "pilot-b", "title": "Pilot B", "users": 20, "satisfied": 15},
    ],
    "destination": "qualification-research",
    "required_tools": [
        "notion-search",
        "notion-fetch",
        "notion-create-pages",
        "notion-update-page",
    ],
    "expected_writes": 1,
}

DECISIONS = {
    "kind": "configured-mcp-fixture",
    "path": "notion-decision-fixture",
    "database_id": "qualification-decisions",
    "existing_records": [],
    "schema": {
        "title": "title",
        "owner": "text",
        "status": "select:accepted",
        "date": "date",
    },
    "required_tools": [
        "notion-search",
        "notion-fetch",
        "notion-create-pages",
        "notion-update-page",
    ],
    "expected_writes": 1,
}


def _manager(tmp_path: Path, declaration: dict[str, Any]):
    from skillrunner.mcp import MCPManager

    fixture = tmp_path / "declaration.json"
    fixture.write_text(json.dumps(declaration))
    audit = tmp_path / "audit.jsonl"
    executable = str(Path(sys.executable).resolve())
    stat = Path(executable).stat()
    policy = Policy(
        allowed_executables=[executable],
        allowed_env=["TEST_NOTION_FIXTURE_PYTHONPATH"],
        executable_identities={
            executable: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        },
    )
    supervisor = ProcessSupervisor(policy, shutdown_grace=0)
    config = MCPConfig(
        transport="stdio",
        command=executable,
        args=[
            "-m",
            "skillrunner.qualification.notion_fixture",
            "--fixture",
            str(fixture),
            "--audit",
            str(audit),
        ],
        cwd=tmp_path,
        env={"PYTHONPATH": "TEST_NOTION_FIXTURE_PYTHONPATH"},
        allowed_tools=declaration["required_tools"],
    )
    pythonpath = os.pathsep.join(
        [
            str(Path(__file__).parents[2] / "src"),
            *(entry for entry in sys.path if "site-packages" in entry),
        ]
    )
    manager = MCPManager(
        {"notion": config},
        policy,
        supervisor,
        {
            **build_child_environment(os.environ, references={}).values,
            "TEST_NOTION_FIXTURE_PYTHONPATH": pythonpath,
        },
        Deadline(5),
        cwd=tmp_path,
    )
    return manager, supervisor, audit


def _tool(manager: Any, original_name: str) -> str:
    return next(tool.name for tool in manager.tools if tool.original_name == original_name)


def _payload(result: dict[str, Any]) -> Any:
    assert result.get("isError", False) is False, result
    return json.loads(result["content"][0]["text"])


@pytest.mark.parametrize(
    ("declaration", "query", "expected_ids"),
    [
        (RESEARCH, "Pilot", ["pilot-a", "pilot-b"]),
        (DECISIONS, "Atlas CSV export", []),
    ],
)
async def test_search_and_fetch_only_declared_records(
    tmp_path: Path,
    declaration: dict[str, Any],
    query: str,
    expected_ids: list[str],
):
    manager, supervisor, _ = _manager(tmp_path, declaration)
    await manager.connect()
    try:
        assert {tool.original_name for tool in manager.tools} == set(declaration["required_tools"])
        create_tool = next(
            tool for tool in manager.tools if tool.original_name == "notion-create-pages"
        )
        destination_key = "database_id" if "database_id" in declaration else "destination"
        assert declaration[destination_key] in create_tool.description
        for property_name, rule in declaration.get("schema", {}).items():
            assert property_name in create_tool.description
            assert rule in create_tool.description
        search = _payload(await manager.invoke(_tool(manager, "notion-search"), {"query": query}))
        assert [item["id"] for item in search["results"]] == expected_ids
        assert all(
            item["url"].startswith("https://fixture.invalid/notion/") for item in search["results"]
        )

        for record_id in expected_ids:
            fetched = _payload(
                await manager.invoke(_tool(manager, "notion-fetch"), {"id": record_id})
            )
            assert fetched["id"] == record_id
            assert fetched["url"] == f"https://fixture.invalid/notion/{record_id}"
    finally:
        await manager.aclose()
    assert not supervisor.active


async def test_authorized_decision_create_persists_one_bounded_audit_record(tmp_path: Path):
    manager, supervisor, audit = _manager(tmp_path, DECISIONS)
    await manager.connect()
    try:
        created = _payload(
            await manager.invoke(
                _tool(manager, "notion-create-pages"),
                {
                    "parent": {"data_source_id": "qualification-decisions"},
                    "pages": [
                        {
                            "properties": {
                                "title": "Atlas export format decision",
                                "owner": "Morgan",
                                "status": "accepted",
                                "date": "2026-09-18",
                            },
                            "content": (
                                "Atlas chose CSV export before PDF export because 8 of 10 "
                                "pilot users requested CSV. Audience: engineering."
                            ),
                        }
                    ],
                },
            )
        )
        assert created["pages"] == [
            {
                "id": "fixture-page-0001",
                "url": "https://fixture.invalid/notion/fixture-page-0001",
            }
        ]
    finally:
        await manager.aclose()

    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert records == [
        {
            "sequence": 1,
            "tool": "notion-create-pages",
            "status": "created",
            "destination": "qualification-decisions",
            "page_id": "fixture-page-0001",
            "properties": {
                "title": "Atlas export format decision",
                "owner": "Morgan",
                "status": "accepted",
                "date": "2026-09-18",
            },
            "content": (
                "Atlas chose CSV export before PDF export because 8 of 10 pilot users "
                "requested CSV. Audience: engineering."
            ),
        }
    ]
    assert not supervisor.active


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "notion-create-pages",
            {
                "parent": {"data_source_id": "production-decisions"},
                "pages": [{"properties": {}, "content": "must not be written"}],
            },
        ),
        (
            "notion-update-page",
            {"page_id": "unknown-page", "properties": {"status": "accepted"}},
        ),
        (
            "notion-create-pages",
            {
                "parent": {"data_source_id": "qualification-decisions"},
                "pages": [
                    {
                        "properties": {
                            "title": "Atlas decision",
                            "owner": "Morgan",
                            "status": "accepted",
                            "date": "2026-02-30",
                        },
                        "content": "must not be written",
                    }
                ],
            },
        ),
    ],
)
async def test_invalid_write_target_fails_before_audit_mutation(
    tmp_path: Path, tool_name: str, arguments: dict[str, Any]
):
    manager, supervisor, audit = _manager(tmp_path, DECISIONS)
    await manager.connect()
    try:
        result = await manager.invoke(_tool(manager, tool_name), arguments)
        assert result["isError"] is True
    finally:
        await manager.aclose()
    assert not audit.exists() or audit.read_text() == ""
    assert not supervisor.active
