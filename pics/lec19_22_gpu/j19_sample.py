"""lec19: generate the same prompts and seeds from the base model and from every snapshot of both runs.

results/lec19/samples/base/<key>-<seed>.jpg          step 0 (shared by both runs)
results/lec19/samples/<run>/<step>/<key>-<seed>.jpg  after <step> training steps
"drift" samples of "a photo of a dog" use DB_DRIFT_SEEDS and the key "dogx". Skips images that exist.
"""
from common import (CLASS_PROMPT, DB_DRIFT_SEEDS, DB_PROMPTS, DB_RUNS, DB_SEEDS, DB_SNAPSHOTS, DB_STEP0_EXTRA,
                    NEGATIVE, RESULTS, STEPS, WORK, Log, Timer, gen, save_img, sd_pipe)

log = Log('lec19')
OUT = RESULTS / 'lec19' / 'samples'
JOBS = [(k, p, s) for k, p in DB_PROMPTS for s in DB_SEEDS] + [('dogx', CLASS_PROMPT, s) for s in DB_DRIFT_SEEDS]


def run_jobs(pipe, folder, jobs):
    n = 0
    for key, prompt, seed in jobs:
        f = folder / f'{key}-{seed}.jpg'
        if f.exists():
            continue
        im = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS, generator=gen(seed)).images[0]
        save_img(im, f, quality=90)
        n += 1
    return n


pipe = sd_pipe()
with Timer() as t:
    n = run_jobs(pipe, OUT / 'base', JOBS + [(k, p, s) for k, p in DB_STEP0_EXTRA for s in DB_SEEDS])
if n:
    log.time('sample_base', t.s, n)

for run in DB_RUNS:
    for step in DB_SNAPSHOTS[1:]:
        ck = WORK / f'db-{run}' / f'checkpoint-{step}'
        if not (ck / 'pytorch_lora_weights.safetensors').exists():
            print('missing', ck, '(not trained that far?)')
            continue
        folder = OUT / run / f'{step:04d}'
        if all((folder / f'{k}-{s}.jpg').exists() for k, _, s in JOBS):
            continue
        pipe.load_lora_weights(str(ck))
        with Timer() as t:
            n = run_jobs(pipe, folder, JOBS)
        pipe.unload_lora_weights()
        log.time(f'sample_{run}_{step}', t.s, n)
        print(run, step, 'done', flush=True)
