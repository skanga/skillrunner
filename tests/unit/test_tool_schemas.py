from skillrunner.tools.schemas import RunCommandArgs


def test_run_command_working_directory_is_optional():
    arguments = RunCommandArgs(executable="python")
    assert arguments.cwd is None
