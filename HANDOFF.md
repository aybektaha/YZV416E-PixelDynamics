# Handoff — Core Pipeline Ready (Aybek)

Hi Emre & Omer 👋

The **core pipeline is done and validated end-to-end on Apple Silicon (MPS)**:
RAFT optical flow → flow-based region growing → DAVIS **J&F** evaluation, plus an
ablation harness. You can now plug your parts in **without touching my code** —
just produce files in the agreed formats below and reuse `evaluate.py`.

---

## 1. What works right now

```bash
# from repo root, with the venv active
python scripts/run_pipeline.py --sequences bear        # flow -> RG -> J&F
```

Validated numbers on `bear` (RAFT + region growing):

| setting | J | F | J&F |
|---|---|---|---|
| default (camera comp ON) | 0.557 | 0.548 | **0.552** |
| camera comp OFF | 0.083 | 0.149 | 0.116 |

So camera-motion compensation alone moves J&F from 0.12 → 0.55 on `bear`. The full
ablation grid is in `scripts/run_ablations.py` (writes `results/ablations.csv`).

Scripts (all in `scripts/`):
- `flow_extraction.py` — RAFT flow, device auto (cuda/mps/cpu), saves `.npy`
- `region_growing.py` — motion region growing, every ablation knob is a CLI flag
- `metrics.py` / `evaluate.py` — DAVIS J & F (multi-instance Hungarian matching)
- `run_pipeline.py` — one-command flow→RG→eval
- `run_ablations.py` — ablation grid → CSV

---

## 1b. Dataset — use the EXACT same one (so sequences & GT match)

We use **only one dataset: DAVIS** (as committed in our proposal's Data Collection
Plan — SegTrack v2 was just a template alternative, we did not pick it). Get the
identical copy via the repo script (official ETH Zürich mirror):

```bash
bash scripts/download_davis.sh
# downloads & extracts:
#   https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip
# -> data/DAVIS/  (90 seqs: 60 train + 30 val, 480p, unsupervised annotations)
```

Note: the file is named "DAVIS-2017-Unsupervised-trainval" but this **is** the
DAVIS 2020 unsupervised trainval split (same 60/30 sequences) — just mention this
in the report to avoid confusion. We evaluate on the **val split (30, with GT)**.
Do NOT use a different DAVIS download (e.g. the 2017 *semi-supervised* set) — the
annotations differ and J&F would not be comparable.

## 2. The shared contracts (please follow exactly)

**(A) Optical flow** — one `.npy` per frame, shape `(H, W, 2)`, `float32`:
```
data/flow/<backbone>/<sequence>/<frame>.npy
# <backbone> ∈ {raft, gmflow, flowformer}
# <frame>.npy = flow from <frame> to <frame>+1
# last frame: copy the previous frame's flow (keeps counts aligned with GT)
```

**(B) Segmentation masks** — DAVIS-palette PNG, pixel value = instance id (0 = bg):
```
results/<method>/<sequence>/<frame>.png
# use region_growing.save_mask() so the palette matches DAVIS
```

**(C) Evaluation** — everyone runs the SAME command so numbers are comparable:
```bash
python scripts/evaluate.py --pred results/<method> --sequences val --csv results/<method>.csv
```

---

## 3. Emre — GMFlow + SAM baseline

**GMFlow:** clone GMFlow, run it on DAVIS, and write flow into contract (A) at
`data/flow/gmflow/<seq>/<frame>.npy`. Easiest path: copy `flow_extraction.py`,
swap the model load + forward call, keep the save/loop logic identical. Then:
```bash
python scripts/region_growing.py --flow-root data/flow/gmflow --out-root results/gmflow_rg --sequences val
python scripts/evaluate.py --pred results/gmflow_rg --sequences val --csv results/gmflow_rg.csv
```
GMFlow may be heavier — if MPS struggles, run it on Colab and copy the `.npy`s back.

**SAM baseline (appearance-only):** run SAM2/SAM3 per frame (no flow), write
instance masks into contract (B) at `results/sam/<seq>/<frame>.png`, then evaluate
with the same `evaluate.py`. This is our motion-vs-appearance comparison — the
whole point is to show how much the motion signal adds over appearance-only.
Use `region_growing.save_mask(labels, path)` for the palette.

## 4. Omer — FlowFormer + backbone comparison (+ optional unified)

**FlowFormer:** same as GMFlow — produce `data/flow/flowformer/<seq>/<frame>.npy`,
then region-grow + evaluate into `results/flowformer_rg`.

**Backbone comparison:** once raft/gmflow/flowformer CSVs exist, build the summary
table + bar chart (J, F, J&F per backbone) for the report. All three use the
identical region-growing + eval code, so differences are purely the flow quality.

**Unified framework (optional):** already supported — `region_growing.py` blends
flow + RGB when `--lambda-rgb > 0` (e.g. `--lambda-rgb 1.0 --rgb-tau 12`). Good as
an extra ablation: flow-only vs flow+RGB.

---

## 4b. Can Emre & Omer work in parallel? (mostly YES)

Your tasks are **independent** — neither blocks the other, and both are already
unblocked because the core code (`region_growing.py`, `evaluate.py`, flow format)
is done. Work simultaneously:

| Task | Owner | Depends on | Parallel? |
|---|---|---|---|
| GMFlow flow + RG + eval | Emre | core code (✅) | ✅ independent |
| SAM appearance baseline | Emre | nothing (RGB-only) | ✅ independent |
| RG criterion flow-only vs flow+RGB | Emre | RAFT flow (✅ exists) | ✅ independent |
| FlowFormer flow + RG + eval | Omer | core code (✅) | ✅ independent |
| Unified framework (optional) | Omer | RAFT flow (✅) | ✅ independent |
| **Backbone comparison table/chart** | Omer | **all 3 CSVs** (raft✅ + gmflow=Emre + flowformer=Omer) | ⚠️ waits |

**The only sequential step** is Omer's final backbone-comparison table: it needs
all three `results/*_rg.csv` files. So each of you first produces your own CSV
independently; once Emre's `gmflow_rg.csv` and Omer's `flowformer_rg.csv` exist,
Omer merges the three (with the existing `raft_rg.csv`) into the comparison
table/bar chart. That last merge is only a few minutes of work.

## 5. Division of remaining work (report + demo, per instructor's announcement)

- **Demo video (4–5 min)** + Google Drive link in the report — collaborative.
- **Report (6–8 pages)** — collaborative; each owns their section.
- **Ablations:** Aybek = threshold (fixed/adaptive) + connectivity + smoothing +
  camera comp (done, in `run_ablations.py`); Emre = RG criterion flow-only vs
  flow+RGB + SAM comparison; Omer = backbone comparison (natural ablation).
- **Failure cases:** each person collects 1–2 failure examples from their method.
- Reminder: everyone must be able to answer a question about *their own* part.

Ping me if any contract is unclear. — Aybek
