import importlib
import json

from skillrunner.model.protocol import ModelToolCall


def api():
    assert importlib.util.find_spec("skillrunner.runtime.context") is not None
    return importlib.import_module("skillrunner.runtime.context")


def test_full_context_is_retained_across_turns():
    context = api().RunContext(
        runner_instructions="runner rules",
        prompt="task",
        catalog=[{"name": "one"}],
        input_inventory=[{"path": "inputs/a"}],
    )
    context.activate("one", "complete skill instructions", "/skills/one")
    context.append_public_reply("working")
    context.append_tool_result("call1", {"text": "page", "truncated": True})
    messages = context.messages()
    serialized = json.dumps(messages)
    for text in ("runner rules", "task", "complete skill instructions", "inputs/a", "page"):
        assert text in serialized
    assert context.estimate([]) >= len(serialized) // 2
    assert messages[-1]["role"] == "tool"


def test_caller_mutation_cannot_change_context():
    catalog = [{"name": "one"}]
    context = api().RunContext(runner_instructions="rules", prompt="task", catalog=catalog)
    catalog[0]["name"] = "changed"
    messages = context.messages()
    messages[0]["content"] = "changed"
    assert "changed" not in json.dumps(context.messages())


def test_repeated_activation_preserves_original_snapshot():
    context = api().RunContext(runner_instructions="rules", prompt="task", catalog=[])
    context.activate("one", "original", "/skills/one")
    context.activate("one", "original", "/skills/one")
    assert json.dumps(context.messages()).count("original") == 1


def test_tool_call_history_preserves_ids_and_arguments():
    context = api().RunContext(runner_instructions="rules", prompt="task", catalog=[])
    context.append_public_reply(
        "working", [ModelToolCall("id1", "read_text", {"path": "inputs/a"}, '{"path":"inputs/a"}')]
    )
    message = context.messages()[-1]
    assert message["tool_calls"][0]["id"] == "id1"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "inputs/a"}


def test_changed_activation_is_rejected_without_replacing_instructions():
    import pytest

    context = api().RunContext(runner_instructions="rules", prompt="task", catalog=[])
    context.activate("one", "original", "/skills/one")
    with pytest.raises(ValueError, match="source_changed"):
        context.activate("one", "replacement", "/skills/one")
    assert "replacement" not in json.dumps(context.messages())


def test_long_instructions_and_history_are_never_silently_truncated():
    context = api().RunContext(runner_instructions="rules", prompt="task", catalog=[])
    instructions = "λ" * 100000 + "END-OF-INSTRUCTIONS"
    context.activate("one", instructions, "/skills/one")
    for number in range(50):
        context.append_public_reply(f"turn {number}")
    messages = context.messages()
    resources = json.loads(messages[2]["content"].split("\n", 1)[1])
    assert resources["active_skills"][0]["instructions"] == instructions
    assert [message["content"] for message in messages[3:]] == [f"turn {n}" for n in range(50)]
