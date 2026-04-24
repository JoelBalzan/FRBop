import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


COMMANDS = {
    "dm": ("DM optimization", "src/frbop/dmop/dm_optimisation.py"),
    "rm": ("RM fitting", "src/frbop/rmop/rm_fitting.py"),
    "sc-fit": ("Scintillation fit", "src/frbop/scop/fit_frb_scintillation.py"),
    "sc-pipeline": ("Scintillation pipeline", "src/frbop/scop/Scintillation_pipeline.py"),
    "sn": ("S/N optimization", "src/frbop/snop/snop_cli.py"),
}


def _find_repo_root(start: Path) -> Path:
    env_root = os.environ.get("FRBOP_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "src" / "frbop").exists():
            return candidate

    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "frbop").exists():
            return candidate
    raise RuntimeError(
        "Could not locate FRBop repository root. Run from inside the repository or use editable install."
    )


def _run_script(script_path: Path, forwarded_args: Sequence[str]) -> int:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(script_path), *forwarded_args]
        sys.path.insert(0, str(script_path.parent))
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency while running {script_path.name}: {exc}. "
            "Install the required extras, e.g. pip install -e '.[all]'"
        )
        return 2
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code)
        return 1
    finally:
        sys.argv = old_argv
        sys.path = old_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frbop",
        description="Unified CLI for FRB optimization tools",
        epilog=(
            "Examples:\n"
            "  frbop dm --stokes-i I.npy --freq freq.npy --time time.npy\n"
            "  frbop rm -i I.npy -q Q.npy -u U.npy --freq freq.npy\n"
            "  frbop sc-fit FRB_250607_htr_dsI.npy --freq FRB_250607_htr_freq.npy\n"
            "  frbop sc-pipeline FRB_250607.yaml --input-mode auto\n"
            "  frbop sn -x xpol.npy -y ypol.npy -p parameters.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS.keys()), help="Tool to run")
    return parser


def main() -> int:
    argv = sys.argv[1:]
    parser = _build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = argv[0]
    if command not in COMMANDS:
        parser.error(f"invalid choice: '{command}' (choose from {', '.join(sorted(COMMANDS.keys()))})")

    remaining = argv[1:]
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    repo_root = _find_repo_root(Path.cwd())
    script_rel_path = COMMANDS[command][1]
    script_path = repo_root / script_rel_path
    return _run_script(script_path, remaining)
