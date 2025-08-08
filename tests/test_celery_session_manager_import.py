import pathlib
import subprocess
import sys
import textwrap


def test_tasks_can_import_session_manager(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        f"""
        import os, sys, importlib.util
        os.chdir({repr(str(tmp_path))})
        # Remove project root from sys.path to simulate running from elsewhere
        sys.path = [p for p in sys.path if p != {repr(str(root))}]
        import types
        orchestrators_stub = types.ModuleType('orchestrators')
        orchestrators_stub.run_credit_repair_process = lambda *a, **k: None
        orchestrators_stub.extract_problematic_accounts_from_report = lambda *a, **k: None
        sys.modules['orchestrators'] = orchestrators_stub
        spec = importlib.util.spec_from_file_location('tasks', {repr(str(root / 'tasks.py'))})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import session_manager
        print(session_manager.__file__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "session_manager.py" in result.stdout
