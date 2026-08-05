#!/usr/bin/env python3
"""Run the whole suite in one command, with or without pytest installed.

    python3 tools/run_tests.py            # everything
    python3 tools/run_tests.py backtest   # only modules matching a substring

Two kinds of test module live in tests/: script-style (a `__main__` block that
calls its own functions) and pytest-style. Both are executed here, so "the
suite passed" means the same thing regardless of which style a file uses.
When pytest is genuinely installed it is used; otherwise tools/pytest_shim
supplies the three symbols the pytest-style modules need.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def _uses_pytest(p: Path) -> bool:
    return "import pytest" in p.read_text()


def main(argv: list[str]) -> int:
    needle = argv[0] if argv else ""
    files = sorted(f for f in TESTS.glob("test_*.py") if needle in f.name)
    if not files:
        print(f"no test modules match {needle!r}")
        return 1

    script_style = [f for f in files if not _uses_pytest(f)]
    pytest_style = [f for f in files if _uses_pytest(f)]
    bad = []

    for f in script_style:
        r = subprocess.run([sys.executable, str(f)], cwd=ROOT,
                           capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'} {f.name}")
        if not ok:
            bad.append(f.name)
            print((r.stdout + r.stderr)[-2000:])

    if pytest_style:
        sys.path.insert(0, str(ROOT / "tools" / "pytest_shim"))
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(TESTS))
        import pytest                                   # shim or the real thing
        shimmed = not hasattr(pytest, "__version__")
        print(f"\npytest-style modules ({'shim' if shimmed else 'real pytest'}):")
        mods = []
        for f in pytest_style:
            try:
                mods.append(importlib.import_module(f.stem))
            except Exception as e:
                bad.append(f.name)
                print(f"  FAIL {f.name} (import): {e!r}")
        if shimmed:
            if pytest.main(mods):
                # blame only the modules that actually failed, never the batch
                bad.extend(sorted({f.split("::")[0] + ".py"
                                   for f in getattr(pytest.main, "failed", [])}))
        else:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                *[str(f) for f in pytest_style]], cwd=ROOT)
            if r.returncode:
                bad.append("pytest run")

    print(f"\n{len(files) - len(set(bad))}/{len(files)} modules OK")
    if bad:
        print("failing:", ", ".join(sorted(set(bad))))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
