# Lectures 19–22 GPU jobs

Everything the lecture 19–22 slides need from a GPU, as one resumable bundle:

- **lec19**: DreamBooth (LoRA) on five photos of one dog, trained twice (with and without prior preservation),
  sampled at seven points during training, plus similarity measurements.
- **lec20**: image-to-image strength sweeps; inpainting with ordinary SD v1.5 and with the inpainting model.
- **lec21**: prompt weighting, semantic guidance (SEGA), inversion (DDIM and LEDITS++), InstructPix2Pix.
- **lec22**: IP-Adapter, ControlNet (edges, depth, pose), and both together.

Everything is Stable Diffusion v1.5-sized, in 16-bit floats, and should fit an **8 GB NVIDIA GPU**.
Downloads (about 20 GB, public models, no login) go to the Hugging Face cache; set `HF_HOME` to put them
somewhere roomy. Nothing is uploaded.

## Running it

```bash
unzip lec19_22_gpu.zip && cd lec19_22_gpu
./setup.sh                         # pip installs; keeps the machine's own PyTorch
L19_QUICK=1 ./run_all.sh           # smoke test, about 10 min plus downloads: every job, very few steps
tmux new -s lec19                  # or nohup, so a dropped connection doesn't stop the real run
./run_all.sh 2>&1 | tee run.log    # the real run
```

The smoke test trains at full size (512 px, a class image beside each subject image) for 20 steps, so if it
passes, the real training fits in the card's memory.

Every job skips work already done, so if anything stops, run `./run_all.sh` again. A failed job doesn't stop
the later ones: `results/status.txt` lists what worked.

Please send back **`lec19_22_results.zip`** (made at the end, roughly 100–200 MB). If the smoke test shows
a failure, sending `lec19_22_results_quick.zip` (logs included) is the quickest way to get it fixed.

## Expected time

Measured speeds go into `results/*-log.json`. Rough guesses for a mid-range 8 GB card:

| Job | What | Time |
|---|---|---|
| `j00_probe` | one timed image | 2 min |
| `j19_train` | 200 class images, then 2 × 1,600 training steps | 1–2 h |
| `j19_sample` | ~300 images from 13 snapshots | 15–30 min |
| `j19_measure` | DINO/CLIP similarities, loss, LoRA size | 3 min |
| `j20_img2img` | ~60 images | 10 min |
| `j21_editing` | ~150 images, two inversions | 30–40 min |
| `j22_control` | ~70 images | 15–20 min |

To shorten lecture 19's training, `L19_STEPS=800 ./run_all.sh` stops each run at 800 steps.

## Notes

- Training uses the unmodified official `train_dreambooth_lora.py` from `diffusers` v0.36.0 (included),
  run with `accelerate launch`. The settings (prompts, seeds, steps, learning rate) are all in `common.py`.
- GPU memory during training is logged every 2 s with `nvidia-smi` into `results/lec19/memory-*.csv`.
- `inputs/` holds the input images: the DreamBooth dataset's `dog` photos (Unsplash, by @alvannee), scikit-image
  sample photos, two public-domain artworks from the Cleveland Museum of Art, and a sketch.
- The safety checker is switched off; every output is reviewed by hand before it goes on a slide.

## Lecture 21 retune (interactive)

`lec21_retune.ipynb` redoes the SEGA and LEDITS++ examples interactively on the Hub GPU: grids to choose a street
prompt and seeds with parked cars, and LEDITS++ edit scale/threshold/skip sweeps, then cells that save the chosen
set. Upload `../lec21_retune.zip` (the notebook, `common.py`, `inputs/cat.png` and the current `info.json`),
unzip it, run the notebook a section at a time, and bring back `lec21_retune_results.zip`. To change the notebook,
edit `lec21_retune_build.py` and run `python lec21_retune_build.py lec21_retune.ipynb`.

Bringing the results back (the zip holds `results-lec21/lec21/...`, not `results/lec21/...`):

1. Delete `pics/lec19_22_raw/results/lec21/sega/`, so that no old seed-21/22 images are packed into the demo.
2. Copy `results-lec21/lec21/*` over `pics/lec19_22_raw/results/lec21/` (new `sega/`, `invert/ledits-*.jpg`,
   and `info.json` with updated `sega` and `ledits` keys; the other keys are kept).
3. From `notes/slides`: `python pics/lec19_22_figs.py`. The SEGA demos read the seeds and scales from the data.
4. Rewrite the `lec21.qmd` text the new images change: "Our street had almost no cars", the threshold bullet about
   the road and houses, the SEGA prompt if it changed, "too gentle… only the nose", and the LEDITS++ captions and
   alt texts (and the settings, if the slide should quote them).
