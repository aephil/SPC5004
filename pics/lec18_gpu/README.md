# Lecture 18 GPU jobs

Everything the lecture-18 slides need from a GPU: a memory probe, then a real full fine-tune of Stable
Diffusion v1.5's UNet on about 450 public-domain Japanese woodblock prints, with snapshots along the way.

## Suggested machine

| | Suggested | Minimum |
|---|---|---|
| GPU | **1 ×** NVIDIA A100 (40 or 80 GB), H100, or L40S (48 GB) | 1 × 24 GB (RTX 4090, A10, L4) |
| vCPU | 8 | 4 |
| RAM | 32 GB | 24 GB |
| Disk | 100 GB, on the persistent volume | 60 GB |

**One GPU is enough.** The training script runs as a single process, and at this size several GPUs would only
add set-up work. The scripts read the card's memory and choose the batch split, gradient checkpointing and
8-bit Adam to suit (see `recipe()` in `common.py`). The effective batch is always 4.

Disk use: about 12 GB of models in the Hugging Face cache (SD v1.5 in fp32 and fp16, BLIP, CLIP), 1.5 GB of
images, at most two training checkpoints of 10–15 GB each, and the final 5 GB pipeline. If the machine has a
separate persistent volume, unzip there and set `export HF_HOME=/that/volume/hf` so nothing large lands on
a temporary disk.

## Running it

```bash
unzip lec18_gpu.zip && cd lec18_gpu
./setup.sh                 # pip installs; keeps the machine's own PyTorch
tmux new -s lec18          # or screen / nohup, so a dropped connection doesn't stop it
./run_all.sh 2>&1 | tee run.log
```

Every step is resumable. If anything stops (a crash, a dropped session, a preempted instance), run
`./run_all.sh` again and it carries on from where it got to.

When it finishes, send back **`lec18_results.zip`** (made by the last step, roughly 100–200 MB).

## Expected time

| Step | A100 / H100 | 24 GB card |
|---|---|---|
| `probe_memory.py`: 14 memory configurations | 20 min | 25 min |
| `prepare_data.py`: download and caption | 10 min | 10 min |
| `train_and_sample.py`: 6,400 steps, 9 snapshots | 1–1.5 h | 2–3 h |
| `collect.py` | 5 min | 5 min |

These are estimates. The stages double in length (50, 100, 200, … 6,400 steps), so stopping early still
leaves a usable sequence: `LEC18_STAGES=50,100,200,400,800,1600,3200 ./run_all.sh` stops at 3,200 steps.

## What each step does

- **`probe_memory.py`**: trains the real UNet for 10 steps in each of 14 configurations (fp32 or fp16
  mixed precision, gradient checkpointing, AdamW or 8-bit Adam, EMA on the GPU, in CPU RAM or none, micro-batch
  1 or 4). Records peak memory, a breakdown and seconds per step. Running out of memory is recorded as a
  result. Writes `results/probe.json`.
- **`prepare_data.py`**: lists the Cleveland Museum of Art's open-access (CC0) Japanese woodblock prints
  (488 after dropping very tall or wide ones, and one partly nude bathing scene) and downloads them. It keeps 40 aside, never trained on, and
  captions every print with BLIP. Writes an `imagefolder` dataset with `metadata.jsonl`, plus credits,
  thumbnails and examples for the slides.
- **`train_and_sample.py`**: runs the unmodified official `diffusers` script `train_text_to_image.py` (v0.36.0,
  included) with `accelerate launch` in stages. After each stage it generates 9 prompts × 2 seeds with both the
  EMA and the raw weights, and measures the loss at fixed noise on training and held-out prints. Two of the
  prompts are training captions, to check for memorisation.
- **`collect.py`**: extracts the logged training loss, finds each generated image's nearest training prints
  (CLIP image similarity), and zips `results/`.

Nothing is uploaded. Downloads are from the Hugging Face Hub (public models; no login needed) and the
Cleveland Museum of Art's open-access API.

The settings (prompts, seeds, stages, learning rate) are all in `common.py`.
