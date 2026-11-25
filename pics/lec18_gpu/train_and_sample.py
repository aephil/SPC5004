"""Fine-tune Stable Diffusion v1.5's UNet on the woodblock prints in stages, sampling after each stage.

    python train_and_sample.py

Each stage runs the official diffusers script (train_text_to_image.py, v0.36.0, unmodified) with
`accelerate launch`, resuming from the previous stage's checkpoint and stopping at the next step count in
STAGES. Then sample_stage.py generates the fixed prompts and measures the fixed-noise loss.

Resumable at every level: a finished stage is skipped, and a stage interrupted mid-training restarts from the
last checkpoint. Older checkpoints are deleted once sampled (each is 10-15 GB); the latest is kept to resume
from. Records the settings and timings in results/run_info.json and the training logs in results/logs/.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import BASE, DATA, RESULTS, RUN, STAGES, gpu_info, have_bnb, recipe, save_json

HERE = Path(__file__).resolve().parent


def train_cmd(r, step):
    cmd = ['accelerate', 'launch', '--num_processes', '1', '--num_machines', '1',
           '--mixed_precision', r['mixed_precision'], '--dynamo_backend', 'no',
           str(HERE / 'train_text_to_image.py'),
           f'--pretrained_model_name_or_path={BASE}',
           f'--train_data_dir={DATA / "train"}', '--image_column=image', '--caption_column=text',
           '--resolution=512', '--center_crop', '--random_flip',
           f'--train_batch_size={r["train_batch_size"]}',
           f'--gradient_accumulation_steps={r["gradient_accumulation_steps"]}',
           f'--mixed_precision={r["mixed_precision"]}', '--use_ema',
           f'--learning_rate={r["learning_rate"]}', '--lr_scheduler=constant', '--lr_warmup_steps=0',
           f'--max_train_steps={step}', f'--checkpointing_steps={step}', '--resume_from_checkpoint=latest',
           f'--output_dir={RUN}', '--report_to=tensorboard', '--seed=0', '--dataloader_num_workers=4']
    if r['gradient_checkpointing']:
        cmd.append('--gradient_checkpointing')
    if r['use_8bit_adam']:
        cmd.append('--use_8bit_adam')
    return cmd


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('$', ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last = 0.0
        for line in p.stdout:                    # text mode turns tqdm's \r redraws into separate lines
            progress = line.startswith('Steps:') or '%|' in line
            if progress and time.time() - last < 30:
                continue                         # log and show progress at most every 30 s
            last = time.time() if progress else last
            f.write(line)
            f.flush()
            sys.stdout.write(line[-400:])
            sys.stdout.flush()
        if p.wait():
            raise SystemExit(f'failed ({p.returncode}); see {log}')


def checkpoints():
    return sorted((int(d.name.split('-')[1]), d) for d in RUN.glob('checkpoint-*') if d.is_dir())


def main():
    info_path = RESULTS / 'run_info.json'
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if 'recipe' not in info:
        g = gpu_info()
        info.update({'gpu': g, 'recipe': recipe(g['vram_gb'], have_bnb()), 'stages': STAGES,
                     'n_train': sum(1 for _ in open(DATA / 'train' / 'metadata.jsonl')), 'timing': {}})
        save_json(info, info_path)
    r = info['recipe']
    print('GPU:', info['gpu'], '\nrecipe:', r, flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'logs').mkdir(parents=True, exist_ok=True)

    for step in [0] + STAGES:
        done = RESULTS / 'samples' / f's{step:05d}' / 'DONE'
        if done.exists():
            print(f'stage {step}: done already')
            continue
        timing = info['timing'].setdefault(str(step), {})
        if step and step not in dict(checkpoints()):
            t0 = time.time()
            run(train_cmd(r, step), RESULTS / 'logs' / f'train-{step:05d}.log')
            timing['train_s'] = time.time() - t0
            save_json(info, info_path)
        t0 = time.time()
        run([sys.executable, str(HERE / 'sample_stage.py'), str(step)], RESULTS / 'logs' / f'sample-{step:05d}.log')
        timing['sample_s'] = time.time() - t0
        done.write_text('ok\n')
        save_json(info, info_path)
        for s, d in checkpoints():                       # keep only the latest checkpoint, to resume from
            if s < step:
                shutil.rmtree(d)
    print('all stages done')


if __name__ == '__main__':
    main()
