import argparse
import runpy
import sys
from typing import Sequence

COMMANDS = {
    "dm": ("DM optimization", "frbop.dmop.dm_optimisation"),
    "pa": ("PA / RVM fitting", "frbop.paop.paop_cli"),
    "rm": ("RM fitting", "frbop.rmop.rm_fitting"),
    "scint": ("Scintillation fit", "frbop.scop.fit_frb_scintillation"),
    "scatt": ("Scattering timescale fit", "frbop.scop.fit_scattering_timescale"),
    "sn": ("S/N optimization", "frbop.snop.snop_cli"),
}


def _run_module(module_name: str, forwarded_args: Sequence[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *forwarded_args]
        runpy.run_module(module_name, run_name="__main__")
        return 0
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency while running {module_name}: {exc}. "
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frbop",
        description="Unified CLI for FRB optimization tools",
        epilog=(
            "Examples:\n"
            "  frbop dm --stokes-i I.npy --freq freq.npy --time time.npy\n"
            "  frbop pa -q Q.npy -u U.npy --time time.npy --period-ms 1.6\n"
            "  frbop rm -i I.npy -q Q.npy -u U.npy --freq freq.npy\n"
            "  frbop scint FRB_250607_htr_dsI.npy --freq FRB_250607_htr_freq.npy\n"
            "  frbop scatt FRB_250607_htr_dsI.npy --freq FRB_250607_htr_freq.npy\n"
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

    module_name = COMMANDS[command][1]
    return _run_module(module_name, remaining)
