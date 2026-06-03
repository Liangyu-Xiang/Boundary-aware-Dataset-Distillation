#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import swanlab


BEST_RE = re.compile(r"Train Finish! Best accuracy is\s+([0-9.]+)@([0-9]+)")
SUMMARY_RE = re.compile(r"\((\d+)\s+repeats\).*Best, last acc:\s+([0-9.]+)\s+([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an evaluation command and record parsed validation results to swanlab."
    )
    parser.add_argument("--work-dir", required=True, help="Directory to run the evaluation command in.")
    parser.add_argument("--swanlab-project", default="MinimaxDiffusion-CFG-Search")
    parser.add_argument("--swanlab-run", required=True)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ipc", type=int, required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument(
        "--disable-summary-log",
        action="store_true",
        help="Do not log parsed mean/std summary metrics.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Expected evaluation command after '--'.")
    return args


def main():
    args = parse_args()
    if not os.path.isdir(args.work_dir):
        raise FileNotFoundError(args.work_dir)

    run = swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_run,
        config={
            "cfg": args.cfg,
            "temperature": args.temperature,
            "ipc": args.ipc,
            "spec": args.spec,
            "repeat": args.repeat,
            "eval_work_dir": args.work_dir,
            "eval_command": " ".join(args.command),
        },
    )

    repeat_best_acc = []
    repeat_best_epoch = []
    summary_logged = False

    proc = subprocess.Popen(
        args.command,
        cwd=args.work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

        best_match = BEST_RE.search(line)
        if best_match:
            acc = float(best_match.group(1))
            epoch = int(best_match.group(2))
            repeat_idx = len(repeat_best_acc) + 1
            repeat_best_acc.append(acc)
            repeat_best_epoch.append(epoch)
            swanlab.log(
                {
                    "repeat_best_acc": acc,
                    "repeat_best_epoch": epoch,
                    "cfg": args.cfg,
                    "temperature": args.temperature,
                },
                step=repeat_idx,
            )

        summary_match = SUMMARY_RE.search(line)
        if summary_match and not args.disable_summary_log:
            swanlab.log(
                {
                    "best_acc_mean": float(summary_match.group(2)),
                    "best_acc_std": float(summary_match.group(3)),
                    "cfg": args.cfg,
                    "temperature": args.temperature,
                },
                step=args.repeat,
            )
            summary_logged = True

    return_code = proc.wait()

    if repeat_best_acc and not summary_logged and not args.disable_summary_log:
        swanlab.log(
            {
                "best_acc_mean": float(np.mean(repeat_best_acc)),
                "best_acc_std": float(np.std(repeat_best_acc)),
                "best_acc_max": float(np.max(repeat_best_acc)),
                "cfg": args.cfg,
                "temperature": args.temperature,
            },
            step=len(repeat_best_acc),
        )

    swanlab.log(
        {"eval_return_code": return_code, "cfg": args.cfg, "temperature": args.temperature},
        step=args.repeat,
    )
    if hasattr(swanlab, "finish"):
        swanlab.finish()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
