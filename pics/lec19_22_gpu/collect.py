"""Zip results/ (or results-quick/) into lec19_22_results.zip (or ..._quick.zip) to send back."""
import zipfile

from common import HERE, QUICK, RESULTS

out = HERE / ('lec19_22_results_quick.zip' if QUICK else 'lec19_22_results.zip')
n = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RESULTS.rglob('*')):
        if p.is_file() and '.tmp' not in p.name:
            z.write(p, p.relative_to(RESULTS.parent))
            n += 1
print(f'{out.name}: {n} files, {out.stat().st_size / 1e6:.0f} MB')
