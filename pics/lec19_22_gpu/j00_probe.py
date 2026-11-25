"""Quick check that the GPU works, and how fast it is: one SD v1.5 image, timed. ~2 min plus downloads."""
import torch

from common import RESULTS, STEPS, Timer, gen, gpu_info, save_img, save_json, sd_pipe

out = RESULTS / 'probe.json'
info = gpu_info()
print(info)
pipe = sd_pipe()
pipe('warm-up', num_inference_steps=2, generator=gen(0))
torch.cuda.reset_peak_memory_stats()
with Timer() as t:
    im = pipe('a lighthouse on a rocky cliff above the sea at sunset, oil painting', num_inference_steps=STEPS,
              generator=gen(0)).images[0]
info.update(seconds_per_image=round(t.s, 2), steps=STEPS,
            peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
save_img(im, RESULTS / 'probe.jpg')
save_json(info, out)
print(info)
