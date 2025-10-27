import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_review_pack_builder_not_imported_by_runtime_modules() -> None:
    """Ensure runtime entrypoints avoid importing the experimental builder."""

    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        """
        import importlib
        import importlib.util
        import json
        import pathlib
        import sys
        import types

        class _DummyResponse:
            status_code = 200
            headers = {}
            text = ""

        class _DummySession:
            def get(self, *args, **kwargs):
                return _DummyResponse()

            def close(self) -> None:
                pass

        sys.modules.setdefault(
            "requests",
            types.SimpleNamespace(Session=_DummySession, RequestException=Exception),
        )

        modules = ["backend.api.app", "backend.api.tasks"]
        for name in modules:
            importlib.import_module(name)

        pipeline_pkg = types.ModuleType("pipeline")
        pipeline_pkg.__path__ = [str(pathlib.Path("pipeline"))]
        sys.modules["pipeline"] = pipeline_pkg

        spec = importlib.util.spec_from_file_location(
            "pipeline.hooks", pathlib.Path("pipeline/hooks.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["pipeline.hooks"] = module
        spec.loader.exec_module(module)
        setattr(pipeline_pkg, "hooks", module)

        print("RESULT:", json.dumps("backend.frontend.review_pack_builder" in sys.modules))
        """
    )

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    marker: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            marker = line.split("RESULT:", 1)[1].strip()
            break

    assert marker is not None, result.stdout
    assert json.loads(marker) is False
