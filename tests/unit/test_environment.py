import importlib

import pytest


def environment_api():
    assert importlib.util.find_spec("skillrunner.runtime.environment") is not None, (
        "Minimal child environment construction is not implemented"
    )
    return importlib.import_module("skillrunner.runtime.environment")


def test_baseline_excludes_model_and_application_credentials():
    api = environment_api()
    inherited = {
        "PATH": "/usr/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "OPENAI_API_KEY": "model-secret",
        "APPLICATION_TOKEN": "application-secret",
        "PYTHONPATH": "/untrusted",
        "LD_PRELOAD": "/untrusted.so",
    }
    result = api.build_child_environment(inherited, references={}, platform="posix")
    assert result.values == {"PATH": "/usr/bin", "TMPDIR": "/tmp", "LANG": "C.UTF-8"}
    assert inherited["OPENAI_API_KEY"] == "model-secret"


def test_explicit_references_are_scoped_and_repr_safe():
    api = environment_api()
    source = {"PRIVATE_TOKEN": "sensitive-value", "OTHER_TOKEN": "unrelated"}
    result = api.build_child_environment(
        source, references={"APP_TOKEN": "PRIVATE_TOKEN"}, platform="posix"
    )
    assert result.values == {"APP_TOKEN": "sensitive-value"}
    assert result.references == {"APP_TOKEN": "PRIVATE_TOKEN"}
    assert "sensitive-value" not in repr(result)
    other = api.build_child_environment(source, references={}, platform="posix")
    assert other.values == {}


def test_missing_reference_reports_name_without_values():
    api = environment_api()
    with pytest.raises(ValueError, match="missing_credential") as caught:
        api.build_child_environment({}, references={"TOKEN": "ABSENT"}, platform="posix")
    assert "ABSENT" in str(caught.value)


def test_windows_baseline_uses_case_insensitive_names():
    api = environment_api()
    result = api.build_child_environment(
        {"Path": "C:\\Windows", "SystemRoot": "C:\\Windows", "TEMP": "C:\\Temp", "KEY": "x"},
        references={},
        platform="nt",
    )
    assert result.values == {"PATH": "C:\\Windows", "SYSTEMROOT": "C:\\Windows", "TEMP": "C:\\Temp"}


@pytest.mark.parametrize("bad_name", ["NAME=VALUE", "NUL\x00NAME", "", "bad-name"])
def test_invalid_reference_names_are_rejected(bad_name):
    api = environment_api()
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.build_child_environment({}, references={bad_name: "SOURCE"}, platform="posix")


def test_nul_secret_fails_without_disclosing_it():
    api = environment_api()
    with pytest.raises(ValueError) as caught:
        api.build_child_environment(
            {"SOURCE": "secret\x00suffix"}, references={"TOKEN": "SOURCE"}, platform="posix"
        )
    assert "secret" not in str(caught.value)


def test_windows_duplicate_destinations_fail_before_launch():
    api = environment_api()
    with pytest.raises(ValueError, match="invalid_configuration"):
        api.build_child_environment(
            {"A": "first", "B": "second"},
            references={"Token": "A", "TOKEN": "B"},
            platform="nt",
        )
