from __future__ import annotations

import argparse

from . import control as control_cli
from . import observe as observe_cli
from . import stress as stress_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime Lab unified runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    observe_parser = sub.add_parser("observe", help="Passive observability run")
    observe_cli.add_observe_args(observe_parser)

    stress_parser = sub.add_parser("stress", help="Branchpoint intervention stress test")
    stress_cli.add_stress_args(stress_parser)

    control_parser = sub.add_parser("control", help="Closed-loop adaptive control run")
    control_cli.add_control_args(control_parser)

    args = parser.parse_args()

    if args.mode == "observe":
        observe_cli.run_from_args(args)
    elif args.mode == "stress":
        stress_cli.run_from_args(args)
    elif args.mode == "control":
        control_cli.run_from_args(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
