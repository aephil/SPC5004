"""lec19: DreamBooth (LoRA) on five photos of one dog, twice: with and without prior preservation.

1. Generates N_CLASS class images ("a photo of a dog") with the untouched model, for prior preservation.
2. Runs diffusers' own train_dreambooth_lora.py (v0.36.0, included) twice with `accelerate launch`, saving a
   checkpoint every DB_CHECKPOINT_EVERY steps so that j19_sample.py can generate from each stage.
Resumable: existing class images are kept, and a stopped run resumes from its latest checkpoint.
"""
import shutil
import subprocess
import sys
import time

from common import (BASE, CLASS_PROMPT, DB_CHECKPOINT_EVERY, DB_RUNS, DB_SETTINGS, DB_STEPS, INPUTS,
                    INSTANCE_PROMPT, N_CLASS, RESULTS, STEPS, WORK, Log, Timer, free, gen, save_img, sd_pipe)

log = Log('lec19')
CLASS_DIR = WORK / 'class'
CLASS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS / 'lec19').mkdir(parents=True, exist_ok=True)

# 1. Class images from the original model
todo = [i for i in range(N_CLASS) if not (CLASS_DIR / f'{i:03d}.jpg').exists()]
if todo:
    pipe = sd_pipe()
    with Timer() as t:
        for i in todo:
            im = pipe(CLASS_PROMPT, num_inference_steps=STEPS, generator=gen(1000 + i)).images[0]
            save_img(im, CLASS_DIR / f'{i:03d}.jpg')
            if i < 24:
                save_img(im.resize((256, 256)), RESULTS / 'lec19' / 'class' / f'{i:03d}.jpg', quality=88)
            print(f'class image {i + 1}/{N_CLASS}', flush=True)
    log.time('class_images', t.s, len(todo))
    del pipe
    free()

# 2. Two training runs
for run in DB_RUNS:
    out = WORK / f'db-{run}'
    if (out / 'pytorch_lora_weights.safetensors').exists():
        print(run, 'already trained')
        continue
    args = [
        'accelerate', 'launch', '--num_processes=1', '--num_machines=1', '--mixed_precision=fp16',
        '--dynamo_backend=no', 'train_dreambooth_lora.py',
        f'--pretrained_model_name_or_path={BASE}',
        f'--instance_data_dir={INPUTS / "dog"}',
        f'--instance_prompt={INSTANCE_PROMPT}',
        f'--output_dir={out}',
        '--resolution=512', '--train_batch_size=1', '--gradient_accumulation_steps=1',
        '--gradient_checkpointing', '--mixed_precision=fp16',
        f'--learning_rate={DB_SETTINGS["learning_rate"]}', '--lr_scheduler=constant', '--lr_warmup_steps=0',
        f'--rank={DB_SETTINGS["rank"]}',
        f'--max_train_steps={DB_STEPS}', f'--checkpointing_steps={DB_CHECKPOINT_EVERY}',
        '--seed=0', '--report_to=tensorboard', '--dataloader_num_workers=0',
    ]
    if DB_SETTINGS['train_text_encoder']:
        args.append('--train_text_encoder')
    if run == 'prior':
        args += ['--with_prior_preservation', f'--class_data_dir={CLASS_DIR}', f'--class_prompt={CLASS_PROMPT}',
                 f'--num_class_images={N_CLASS}', '--prior_loss_weight=1.0']
    if any(out.glob('checkpoint-*')):
        args.append('--resume_from_checkpoint=latest')

    # Log GPU memory every 2 s while training (NVIDIA only; skipped quietly elsewhere)
    mem = None
    if shutil.which('nvidia-smi'):
        f = open(RESULTS / 'lec19' / f'memory-{run}.csv', 'a')
        mem = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.total',
                                '--format=csv,noheader', '-l', '2'], stdout=f)
    print(' '.join(args), flush=True)
    t0 = time.time()
    try:
        code = subprocess.call(args)
    finally:
        if mem:
            mem.terminate()
    if code:
        sys.exit(f'training run {run} failed with exit code {code}')
    log.time(f'train_{run}', time.time() - t0, DB_STEPS)
