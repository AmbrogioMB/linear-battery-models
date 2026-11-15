"""Cross-platform runner. It executes the commands listed in a job file
(jobs/*.txt) in parallel (one process per job, each using GRB_THREADS Gurobi
threads), writes one log per job in logs/, and marks completed jobs in
logs/done/, so that it can be stopped and restarted at any time.

Usage:
  python code/run_jobs.py --jobs jobs/headroom.txt --workers 100 [--threads 1] [--only PATTERN]

Commands are run from the experiment folder. Lines starting with '#' are ignored; a line starting with
'@barrier' waits until all previous jobs have finished.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiment folder
LOGS = os.path.join(HERE, "logs")
DONE = os.path.join(LOGS, "done")


def job_id(cmd):
    name = "_".join(cmd.split()[1:])[:60].replace("/", "-")
    return f"{name}_{hashlib.md5(cmd.encode()).hexdigest()[:6]}"


def run(cmd, threads):
    jid = job_id(cmd)
    marker = os.path.join(DONE, jid)
    if os.path.isfile(marker):
        return jid, "skipped"
    env = dict(os.environ, GRB_THREADS=str(threads), PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(os.path.join(LOGS, jid + ".log"), "a") as log:
        log.write(f"$ {cmd}\n")
        log.flush()
        # commands are written as "python ..."; use the interpreter running this script
        parts = cmd.split()
        if parts[0] == "python":
            parts[0] = sys.executable
        rc = subprocess.call(parts, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    status = "ok" if rc == 0 else f"FAILED (exit {rc})"
    if rc == 0:
        open(marker, "w").write(f"{time.time() - t0:.0f}s\n")
    print(f"[{time.strftime('%H:%M:%S')}] {status:>16}  {time.time() - t0:8.0f}s  {cmd}", flush=True)
    return jid, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--only", default=None, help="run only jobs containing this string")
    a = ap.parse_args()
    os.makedirs(DONE, exist_ok=True)
    blocks, cur = [], []
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@barrier"):
            blocks.append(cur)
            cur = []
        elif a.only is None or a.only in line:
            cur.append(line)
    blocks.append(cur)
    failed = []
    for b in blocks:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for jid, st in ex.map(lambda c: run(c, a.threads), b):
                if st.startswith("FAILED"):
                    failed.append(jid)
    print("all jobs processed;", f"{len(failed)} failed: {failed}" if failed else "no failures")


if __name__ == "__main__":
    main()
