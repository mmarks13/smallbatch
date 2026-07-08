# Compiling in the cloud

The compile step is deliberately just **CLI + plain files** — spec in, data
dir in, artifact dir out; no daemon, no database. Any box with a CUDA GPU
works: rsync up, `smallbatch compile`, rsync the ~30MB adapter back.

The split that matters: **labeling runs locally** (it needs your teacher
credentials and is CPU-only); only the GPU-bound compile ships to the cloud.
Generate `data/<fn>/` first, then launch.

## Turnkey: SkyPilot

[`cloud/skypilot.yaml`](../cloud/skypilot.yaml) launches the cheapest
available spot GPU on whatever clouds you've enabled (AWS/GCP/Azure/RunPod/
vast.ai/Lambda...), installs smallbatch, compiles, and lets you pull the
artifact back:

```bash
pip install "skypilot[aws,gcp]"          # pick your clouds; `sky check` to verify
sky launch -c sb cloud/skypilot.yaml --env SPEC=examples/ticket-priority/spec.yaml
rsync -av sb:~/sky_workdir/artifacts/ artifacts/
sky down sb
```

Practical notes from real use:

- **`spec_files` must exist on the remote.** Keep them inside the repo (they
  ship with `workdir`) or mirror absolute paths with `file_mounts`.
- **Artifacts can outlive a dying spot instance.** Every finished compile
  prints a `MANIFEST::<json>` line to the log; SkyPilot retains job logs, so
  results survive even if the instance is reclaimed before you rsync. Grep
  them back with `sky logs sb | grep '^MANIFEST::'`.
- **Sweeps on rented GPUs** work the same way — put `smallbatch sweep
  sweeps/your-sweep.yaml` in `run:`. Consider `HF_HUB_OFFLINE=1` with
  pre-cached weights in `setup:` if the provider's IPs are rate-limited by
  the HF Hub (common on GPU marketplaces).
- **Spot etiquette:** `--retry-until-up` for patience, and always
  `sky down` — a forgotten instance costs more than the compile.

## Manual: any GPU box

```bash
rsync -av --exclude .venv . gpubox:smallbatch/
ssh gpubox 'cd smallbatch && pip install -e . && smallbatch compile examples/ticket-priority/spec.yaml'
rsync -av gpubox:smallbatch/artifacts/ artifacts/
```

There is no orchestration inside smallbatch itself — that's a feature: the
artifact directory is the entire interface between machines.
