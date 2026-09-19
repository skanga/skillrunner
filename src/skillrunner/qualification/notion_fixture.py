"""Bounded local MCP fixture for the approved Notion qualification cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

_URL_ROOT = "https://fixture.invalid/notion/"
_TOOLS = {
    "notion-search",
    "notion-fetch",
    "notion-create-pages",
    "notion-update-page",
}
_DESTINATIONS = {
    "notion-research-fixture": ("destination", "qualification-research"),
    "notion-decision-fixture": ("database_id", "qualification-decisions"),
}


class _StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CreateParent(_StrictArguments):
    page_id: str | None = Field(
        default=None, description="Approved parent page ID for a document fixture."
    )
    data_source_id: str | None = Field(
        default=None, description="Approved database ID for a structured fixture."
    )


class CreatePage(_StrictArguments):
    properties: dict[str, Any] = Field(
        description="Page properties; database fixtures require the declared schema exactly."
    )
    content: str = Field(description="Nonempty Markdown page content.", min_length=1)


@dataclass
class Fixture:
    records: dict[str, dict[str, Any]]
    destination: str
    destination_kind: str
    schema: dict[str, str]
    expected_writes: int
    audit_path: Path
    writes: int = 0

    @classmethod
    def load(cls, fixture_path: Path, audit_path: Path) -> Fixture:
        raw = json.loads(fixture_path.read_text())
        if raw.get("kind") != "configured-mcp-fixture":
            raise ValueError("Fixture must be a configured-mcp-fixture declaration")
        if set(raw.get("required_tools", ())) != _TOOLS:
            raise ValueError("Fixture must declare exactly the four supported Notion tools")

        fixture_name = raw.get("path")
        if fixture_name not in _DESTINATIONS:
            raise ValueError("Fixture is not one of the two approved Notion qualification cases")
        destination_key, approved_destination = _DESTINATIONS[fixture_name]
        is_database = destination_key == "database_id"
        records_key = "existing_records" if is_database else "records"
        destination = raw.get(destination_key)
        records = raw.get(records_key)
        expected_writes = raw.get("expected_writes")
        if not isinstance(destination, str) or not destination:
            raise ValueError("Fixture declaration has no approved destination")
        if destination != approved_destination:
            raise ValueError("Fixture declaration changes the approved destination")
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ValueError("Fixture records must be a list of objects")
        if type(expected_writes) is not int or expected_writes < 0:
            raise ValueError("Fixture expected_writes must be a nonnegative integer")
        indexed: dict[str, dict[str, Any]] = {}
        for item in records:
            record_id = item.get("id")
            if not isinstance(record_id, str) or not record_id or record_id in indexed:
                raise ValueError("Every fixture record requires a unique string id")
            indexed[record_id] = dict(item)
        schema = raw.get("schema", {})
        if not isinstance(schema, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in schema.items()
        ):
            raise ValueError("Fixture schema must map property names to string types")
        if audit_path.exists():
            raise ValueError("Audit path already exists; provide an exclusive path")
        return cls(
            records=indexed,
            destination=destination,
            destination_kind="data_source_id" if is_database else "page_id",
            schema=schema,
            expected_writes=expected_writes,
            audit_path=audit_path,
        )

    @staticmethod
    def public(record: dict[str, Any]) -> dict[str, Any]:
        value = dict(record)
        value["url"] = _URL_ROOT + str(record["id"])
        return value

    def search(self, query: str) -> dict[str, Any]:
        terms = query.casefold().split()
        results = []
        for record in self.records.values():
            haystack = json.dumps(record, sort_keys=True).casefold()
            if not terms or any(term in haystack for term in terms):
                results.append(self.public(record))
        return {"results": results}

    def fetch(self, record_id: str) -> dict[str, Any]:
        record = self.records.get(record_id)
        if record is None:
            raise ValueError("Unknown fixture page id")
        return self.public(record)

    def _check_write_available(self) -> None:
        if self.writes >= self.expected_writes:
            raise ValueError("Fixture write limit reached")

    def _check_parent(self, parent: dict[str, str | None]) -> None:
        parent = {key: value for key, value in parent.items() if value is not None}
        if parent != {self.destination_kind: self.destination}:
            raise ValueError("Write destination is not authorized by this fixture")

    def _check_properties(self, properties: dict[str, Any]) -> None:
        if not self.schema:
            return
        if set(properties) != set(self.schema):
            raise ValueError("Page properties do not match the declared database schema")
        for name, rule in self.schema.items():
            value = properties[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"Property {name!r} must be a nonempty string")
            if rule.startswith("select:") and value != rule.partition(":")[2]:
                raise ValueError(f"Property {name!r} has an unsupported select value")
            if rule == "date":
                try:
                    parsed = date.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"Property {name!r} must be a valid YYYY-MM-DD date") from None
                if parsed.isoformat() != value:
                    raise ValueError(f"Property {name!r} must be a valid YYYY-MM-DD date")

    def create(self, parent: dict[str, str | None], pages: list[dict[str, Any]]) -> dict[str, Any]:
        self._check_write_available()
        self._check_parent(parent)
        if len(pages) != 1:
            raise ValueError("This fixture accepts exactly one page per create call")
        page = pages[0]
        if set(page) != {"properties", "content"}:
            raise ValueError("A page requires only properties and content")
        properties, content = page["properties"], page["content"]
        if not isinstance(properties, dict) or not isinstance(content, str) or not content:
            raise ValueError("Page properties must be an object and content a nonempty string")
        self._check_properties(properties)

        page_id = f"fixture-page-{self.writes + 1:04d}"
        record = {"id": page_id, "properties": dict(properties), "content": content}
        audit = {
            "sequence": self.writes + 1,
            "tool": "notion-create-pages",
            "status": "created",
            "destination": self.destination,
            "page_id": page_id,
            "properties": dict(properties),
            "content": content,
        }
        self._commit(record, audit)
        return {"pages": [{"id": page_id, "url": _URL_ROOT + page_id}]}

    def update(
        self, page_id: str, properties: dict[str, Any] | None, content: str | None
    ) -> dict[str, Any]:
        self._check_write_available()
        record = self.records.get(page_id)
        if record is None:
            raise ValueError("Unknown fixture page id")
        if properties is None and content is None:
            raise ValueError("Update requires properties or content")
        if properties is not None:
            if not isinstance(properties, dict):
                raise ValueError("Updated properties must be an object")
            merged = dict(record.get("properties", {}))
            merged.update(properties)
            self._check_properties(merged)
        else:
            merged = dict(record.get("properties", {}))
        if content is not None and (not isinstance(content, str) or not content):
            raise ValueError("Updated content must be a nonempty string")
        updated = dict(record)
        updated["properties"] = merged
        if content is not None:
            updated["content"] = content
        audit = {
            "sequence": self.writes + 1,
            "tool": "notion-update-page",
            "status": "updated",
            "destination": self.destination,
            "page_id": page_id,
            "properties": merged,
            "content": updated.get("content", ""),
        }
        self._commit(updated, audit)
        return {"id": page_id, "url": _URL_ROOT + page_id}

    def _commit(self, record: dict[str, Any], audit: dict[str, Any]) -> None:
        data = json.dumps(audit, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        mode = "x" if self.writes == 0 else "a"
        with self.audit_path.open(mode, encoding="utf-8") as stream:
            stream.write(data + "\n")
        self.records[str(record["id"])] = record
        self.writes += 1


def build_server(fixture: Fixture) -> MCPServer:
    server = MCPServer(
        "skillrunner-notion-qualification-fixture",
        description="Local deterministic Notion fixture for two qualification cases only.",
    )

    @server.tool(name="notion-search")
    def notion_search(query: str) -> dict[str, Any]:
        """Search only records declared in this local fixture using case-insensitive terms."""
        return fixture.search(query)

    @server.tool(name="notion-fetch")
    def notion_fetch(id: str) -> dict[str, Any]:
        """Fetch one declared or fixture-created page by its exact id."""
        return fixture.fetch(id)

    @server.tool(
        name="notion-create-pages",
        description=(
            "Create exactly one page in this local fixture. Authorized parent: "
            + json.dumps({fixture.destination_kind: fixture.destination}, sort_keys=True)
            + '. Each page is {"properties": {...}, "content": "Markdown"}. '
            + "Required database property schema (empty means no required schema): "
            + json.dumps(fixture.schema, sort_keys=True)
            + ". No destination outside this fixture is permitted."
        ),
    )
    def notion_create_pages(parent: CreateParent, pages: list[CreatePage]) -> dict[str, Any]:
        return fixture.create(parent.model_dump(), [page.model_dump() for page in pages])

    @server.tool(name="notion-update-page")
    def notion_update_page(
        page_id: str,
        properties: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        """Update a known fixture page with properties, content, or both."""
        return fixture.update(page_id, properties, content)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    fixture = Fixture.load(args.fixture, args.audit)
    build_server(fixture).run(transport="stdio")


if __name__ == "__main__":
    main()
