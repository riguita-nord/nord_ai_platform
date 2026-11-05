# Nord AI Trainer — local web interface

## Start

Stop the interactive Nord CLI first. From the repository root:

    git pull origin main
    .venv/bin/python -m compileall -q scripts/trainer_web.py
    .venv/bin/python scripts/trainer_web.py

Open http://127.0.0.1:8765 on the same computer. Copy the printed access token
into the page. The browser shows dataset size, current training state, recent
logs, and controls for starting/stopping a training session.

For a **trusted private LAN only**, run:

    NORD_TRAINER_TOKEN='CHOOSE-A-LONG-RANDOM-SECRET' \
      .venv/bin/python scripts/trainer_web.py --host 0.0.0.0 --port 8765

Open http://SERVER_PRIVATE_IP:8765 from another computer. HTTP on a LAN is
**unencrypted**, so tokens and requests can be intercepted by anyone with
network visibility. Use a VPN or TLS reverse proxy for stronger protection.
Do not expose port 8765 directly to the public internet.

## Dataset

The trainer requires training/dataset/train.txt to contain at least 1 MB.
Collect Wikipedia text with scripts/dataset_manager.py or import approved
texts with scripts/import_licensed_texts.py.

## Training

- Start: launches scripts/rebuild_language.py using the selected number of
  updates and text/dialogue ratio.
- Stop: sends SIGINT to the training subprocess; the last completed
  checkpoint remains on disk. Up to 49 unsaved updates may be lost.
- Reset: backs up the current language-v1.pt and initializes new weights.
  Use only for deliberate clean experiments, not to resume training.
- Live console: shows the last 70 log lines, updated every four seconds.

The trainer uses the existing exclusive .nord-instance.lock. Do not run
./nord concurrently. If a training process survives a dashboard restart,
the existing lock prevents conflicting checkpoint writes; the restarted
dashboard cannot control that orphaned process.

## Security and scope

The server binds to 127.0.0.1 by default. Every API request requires a
random token (or NORD_TRAINER_TOKEN). The token is not stored in browser
local storage. This is a single-user development dashboard, not a hardened
multi-tenant SaaS service.

The underlying model remains an eight-layer byte-token Transformer trained
on CPU. The dashboard does not make it fluent by itself. No automatic
internet scraping or unsupervised self-training is performed.

## New dashboard navigation

The browser now starts at a dedicated token login. After login, the sidebar
provides Dashboard, Model, Training, Internet, Datasets, Professor, Evaluation,
Checkpoints, and Settings. Professor examples are saved as approved records in
`~/.nord-brain/trainer-feedback.jsonl` and are **not yet used for gradient updates**.
The evaluation page shows existing training logs; it is **not** a separate
benchmark. Checkpoints are listed but cannot yet be restored from the UI.

## Production release gates

This version has **not** been integration-tested on the target VM and is not
certified production-ready. Before public deployment, implement TLS, hardened
sessions and CSRF protection, persistent job supervision/recovery, log rotation,
resource limits, tests for interrupted jobs and dataset integrity, and a proper
review/import path for Professor examples. Until then restrict to a trusted
private LAN or loopback and take backups of the model directory.
