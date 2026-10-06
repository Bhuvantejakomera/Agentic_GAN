from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_SCRIPTS: Dict[str, Path] = {
    "mnist": PROJECT_ROOT / "training" / "train_mnist_gan.py",
    "fashion": PROJECT_ROOT / "training" / "train_fashion_dcgan.py",
    "celeba": PROJECT_ROOT / "training" / "train_celeba_gan.py",
}

OUTPUT_MAP: Dict[str, Path] = {
    "mnist": PROJECT_ROOT / "outputs" / "mnist_digits",
    "fashion": PROJECT_ROOT / "outputs" / "fashion_clothes",
    "celeba": PROJECT_ROOT / "outputs" / "celeba_faces",
}


def ensure_output_structure() -> None:
    for path in OUTPUT_MAP.values():
        path.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)


def append_agent_log(message: str, log_file: Path | None = None) -> None:
    if log_file is None:
        log_file = PROJECT_ROOT / "outputs" / "logs" / "agent.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def run_training(project: str, forwarded_args: List[str]) -> int:
    script = TRAINING_SCRIPTS[project]
    output_dir = OUTPUT_MAP[project]

    cmd = [sys.executable, str(script), "--output-dir", str(output_dir), *forwarded_args]

    append_agent_log(f"Starting {project} training with output_dir={output_dir}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        append_agent_log(f"Completed {project} training successfully")
    else:
        append_agent_log(f"{project} training failed with code {result.returncode}")

    return result.returncode


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Multi-project GAN + Agent runner")
    parser.add_argument(
        "project",
        choices=["mnist", "fashion", "celeba", "all"],
        help="Which project to run",
    )
    parser.add_argument(
        "--agent-log-demo",
        action="store_true",
        help="Append sample agent decision log lines and exit",
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def write_demo_agent_logs() -> None:
    append_agent_log("Epoch 20: Agent reduced learning rate")
    append_agent_log("Epoch 40: Agent increased batch size")
    append_agent_log("Epoch 75: Agent stopped training")


def main() -> int:
    args, forwarded_args = parse_args()
    ensure_output_structure()

    if args.agent_log_demo:
        write_demo_agent_logs()
        print("Demo agent logs written to outputs/logs/agent.log")
        return 0

    if args.project == "all":
        for project in ["mnist", "fashion", "celeba"]:
            code = run_training(project, forwarded_args)
            if code != 0:
                return code
        return 0

    return run_training(args.project, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
