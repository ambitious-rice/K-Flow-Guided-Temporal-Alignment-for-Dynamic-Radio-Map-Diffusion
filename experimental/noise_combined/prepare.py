"""Freeze three new arms completing a2x2 with the historical noise-map arm."""
import json
from pathlib import Path
import subprocess
from experimental.noise_loss_screen.common import write
from .model import VARIANTS

ROOT = Path('runs/noise_combined_20261010')


def main():
    ROOT.mkdir(exist_ok=False)
    old = Path('runs/noise_paper_injection_20261009/noise_map').resolve()
    protocol = dict(
        goal='Six-GPU W16 comparison of noise-map+multilayer/HWM conditioning and paired stratified noise training',
        variants=list(VARIANTS), gpus=['0,1', '2,3', '4,5'],
        reference=str(old), initialization='All random seed20261009; common parameters must equal historical step0 bitwise; additions use isolated seed20261010',
        train='Full9000 videos,W16; rates discrete uniform1..10%; marginal sigma uniform[0,.09); no clean indicator',
        pairing='Four levels per video/mask, one random level per equal-width sigma stratum in random order; common measurement Gaussian field, diffusion noise and t within each group; measurement noise independent of diffusion noise',
        iid='Exact historical FullData RNG, seeds and samples',
        fairness='Global32 presentations/update for all arms; paired has8 independent video/window draws versus32 for IID; this diversity difference is part of treatment. Compare common steps and measured compute, not only wallclock bests.',
        model='Preserve noise maps in both stems and all original pathways; add sigma/.09 MLP, independent zero-init six-parameter modulation head per DiT block, and sigma embedding to HWM/shared observation/output modulation',
        objective='Same six losses,weight1 each: clean reconstruction,sampled_clean,HWM calibration,equation,obstacle,source. Same two-step differentiable DDIM training; no ranking or sigma remapping.',
        validation='Same immutable tune8/confirm16 bank, masks,observations,DDIM50 and diffusion initial states as reference. Tune only selects model/raw EMA, fixed sigma and stopping. Confirm never used for selection.',
        final='Three tune-selected criteria,full360 cases; two extra diffusion seeds on16 confirm videos for best_usable/fallback best_alignment',
        success='Correct-sigma reconstruction plus gain against tune-selected constant with paired-video confidence intervals; report full sigma matrix,row minima,and common-step comparisons against baseline',
        budget='7h training/validation,8.75h hard total; max40000 updates,early stop after4000 withpatience4; time limit saved as resumable,not convergence',
        caveat='Reuse historical baseline rather than rerun it; verify identical initialization/IID batches and archive environment. Prior validation is exploratory;200video test not used. No claim of full CDM/DiT reproduction.')
    write(ROOT/'protocol.json', protocol)
    for variant, gpus in zip(VARIANTS, protocol['gpus']):
        path = ROOT/variant
        path.mkdir()
        (path/'inputs').symlink_to((old/'inputs').resolve(), target_is_directory=True)
        (path/'model_config.yaml').write_text((old/'model_config.yaml').read_text())
        spec = json.loads((old/'config.json').read_text())
        for key in ['started_at', 'train_deadline', 'hard_deadline']:
            spec.pop(key, None)
        spec.update(variant=variant,gpus=gpus)
        write(path/'config.json', spec)
    write(Path('.agents/runs')/(ROOT.name+'.yaml'), dict(
        goal=protocol['goal'], status='prepared', cwd=str(Path.cwd()),
        environment='/data_p6/fzj/conda/envs/RMDM/bin/python', protocol=protocol,
        source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True),
        output=str(ROOT.resolve()), next='Verify initialization/data/loss,simultaneous2GPU smoke per arm,then launch'))


if __name__=='__main__':
    main()
