#!/usr/bin/env python3
"""Run the suite and real-video smoke tests on the owner's machine, outside Jupyter."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-name", action="append", help="Exact filename to locate under data (repeatable)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    report_dir = root / "work" / ("validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    report_dir.mkdir(parents=True, exist_ok=False)
    if not args.skip_tests:
        with (report_dir / "tests.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        print("Tests:", "PASS" if result.returncode == 0 else "FAIL", report_dir / "tests.log", flush=True)
        if result.returncode:
            return result.returncode
    outcomes = []
    for name in args.video_name or ["test_video2.mp4", "test_video3.mp4", "videoplayback.mp4"]:
        matches = sorted((root / "data").rglob(name))
        if (root / name).is_file():
            matches.append(root / name)
        matches = [p for p in matches if "clipiq" not in p.relative_to(root).parts[:-1]]
        if len(matches) != 1:
            outcomes.append({"video": name, "status": "missing" if not matches else "ambiguous", "matches": [str(p) for p in matches]})
            print(f"{name}: expected one source file, found {len(matches)}", flush=True)
            continue
        source = matches[0]
        command = [sys.executable, "-m", "clipiq", "run", str(source),
                   "--output", str(root / "data/processed/clipiq"), "--top-k", "5", "--max-duration", "60"]
        if (root / "data/processed").exists():
            command += ["--reuse-legacy", str(root / "data/processed")]
        if args.no_speech:
            command += ["--no-speech"]
        print("Processing", source, flush=True)
        log_path = report_dir / f"{source.stem}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        outcomes.append({"video": str(source), "status": "passed" if result.returncode==0 else "failed", "log": str(log_path)})
        print(outcomes[-1]["status"], "—", log_path, flush=True)
    (report_dir / "real_video_results.json").write_text(json.dumps(outcomes, indent=2))
    print("Validation report:", report_dir)
    return 0 if outcomes and all(r["status"]=="passed" for r in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
