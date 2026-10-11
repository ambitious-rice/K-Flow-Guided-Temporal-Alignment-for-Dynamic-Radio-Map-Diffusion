"""Freeze the CPU-only, episode-disjoint adaptive cross-fit pilot bank."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from experimental.paper_evaluation.randomness import generator, materialize

SEED = 20261012
MAIN = Path('/data_p6/fzj/projects/RMDM')
SIGMAS = [0., .03, .05, .09]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def video_ids(value, scenes):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from video_ids(key, scenes)
            yield from video_ids(item, scenes)
    elif isinstance(value, list):
        for item in value:
            yield from video_ids(item, scenes)
    elif isinstance(value, str):
        parts = value.split('/')
        if len(parts) == 3 and parts[0] in scenes and parts[1].startswith('episode_'):
            yield value


def variants():
    def item(name, folds=4, members=4, spatial=False, window=16, prefix_only=False, steps=50):
        return dict(name=name, folds=folds, members=members, spatial=spatial,
                    window=window, prefix_only=prefix_only, steps=steps)
    return [item('w16_f4_m8', members=8), item('w16_f4_m4'),
            item('w16_f3_m4', folds=3), item('w16_spatial3_m4', folds=3, spatial=True),
            item('w4_prefix_f4_m4', window=4, prefix_only=True),
            item('w4_chunks_f4_m4', window=4), item('w16_f4_m4_ddim25', steps=25)]


def prepare(root):
    from rmdm.data import WindowDataset
    reference = MAIN/'runs/noise_combined_20261010/hybrid_paired/inputs/protocol.json'
    source_root = Path(json.loads(reference.read_text())['source_root']).resolve()
    split_file = MAIN/'configs/splits/m20_formal075_clean16_scene_split.json'
    scenes = sorted(json.loads(split_file.read_text())['val'])
    samples = sorted(json.loads((source_root/'index.json').read_text())['samples'],
                     key=lambda s: (s['scene_id'], s['episode_id'], s['tx_id']))
    indices = {f"{s['scene_id']}/{s['episode_id']}/{s['tx_id']}": i for i, s in enumerate(samples)}
    exclusions = sorted(set((MAIN/'runs').rglob('validation_bank.json')) | {
        MAIN/'runs/noise_paper_injection_20261009/expanded100/protocol.json',
        MAIN/'runs/paper_test200_20261006/calibration_bank/bank.json',
        MAIN/'runs/paper_test200_20261006/calibration.json',
        MAIN/'runs/noise_sigma_confirm20_20261011/manifest.json',
    })
    excluded = set()
    for path in exclusions:
        excluded.update(v.rsplit('/', 1)[0] for v in video_ids(json.loads(path.read_text()), scenes))
    manifest = []
    for si, scene in enumerate(scenes):
        candidates = [s for s in samples if s['scene_id'] == scene and
                      f"{scene}/{s['episode_id']}" not in excluded]
        episodes = sorted({s['episode_id'] for s in candidates})
        rng = np.random.default_rng(SEED + si * 100000)
        chosen = sorted(rng.choice(episodes, 6, replace=False).tolist())
        for ei, episode in enumerate(chosen):
            transmitters = [s for s in candidates if s['episode_id'] == episode]
            sample = transmitters[int(rng.integers(len(transmitters)))]
            video = f"{scene}/{episode}/{sample['tx_id']}"
            vi = indices[video]
            manifest.append(dict(role='calibration' if ei < 2 else 'evaluation', video_index=vi,
                                 video_id=video, start=0, scene_id=scene, episode_id=episode,
                                 tx_id=sample['tx_id'], input_file=f'inputs/v{vi:05d}_t00.pt'))
    assert len(manifest) == len({m['video_id'].rsplit('/', 1)[0] for m in manifest}) == 12
    assert sum(m['role'] == 'calibration' for m in manifest) == 4
    assert not {m['video_id'].rsplit('/', 1)[0] for m in manifest} & excluded
    checkpoint = MAIN/'runs/noise_test500_select200_20261011/checkpoint.pt'
    config = MAIN/'runs/noise_trajectory_finetune_20261010/fast20_allparams_from2500/model_config.yaml'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['step'] == 7000 and payload['weights'] == 'model'
    del payload
    protocol = dict(schema='noise-adaptive-pilot-v1', seed=SEED, model_step=7000, checkpoint_step=7000,
                    weights='model', checkpoint_source=str(checkpoint), model_config_source=str(config),
                    source_root=str(source_root), split_file=str(split_file), split='val', rate=2,
                    sigmas=SIGMAS, calibration_sigmas=SIGMAS, initial_sigma=.03, max_rounds=3,
                    qmax=.09**2, starts=[0], window_size=16, manifest=manifest, variants=variants(),
                    videos=[m['video_id'] for m in manifest],
                    full_data_video_indices={m['video_id']: m['video_index'] for m in manifest},
                    excluded_episodes=sorted(excluded), exclusion_sources=[str(p) for p in exclusions],
                    selection='Six uniformly sampled distinct episodes per validation scene, sorted; first two calibration and remaining four evaluation; one random Tx each',
                    calibration_evaluation_disjoint_by_episode=True,
                    estimator='adaptive cross-fit residual variance; not exact EM',
                    calibration_q_nodes=[s*s for s in SIGMAS],
                    seed_rule='seed + full_data_video_index * 1000000 + absolute_frame * 100 + stream',
                    full_data_index_order='sorted complete dataset samples by (scene_id, episode_id, tx_id)',
                    streams=dict(mask=0, measurement_noise=1, initial=2, fold_assignment=3,
                                 ensemble='10 + 8*fold + member'),
                    observations='Y=M*(target+true_sigma*measurement_noise); no RSS clipping; same mask, noise and initial across sigmas',
                    evaluation_scope='New episodes in existing validation scenes; calibration and evaluation are episode-disjoint')
    root.mkdir(parents=True, exist_ok=True)
    path = root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing protocol differs; refusing to overwrite')
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text()) != manifest:
        raise ValueError('Existing manifest differs; refusing to overwrite')
    if (root/'checkpoint.pt').exists() and not os.path.samefile(root/'checkpoint.pt', checkpoint):
        raise ValueError('Existing checkpoint is not the expected hard link; refusing overwrite')
    if (root/'model_config.yaml').exists() and (root/'model_config.yaml').read_bytes() != config.read_bytes():
        raise ValueError('Existing model configuration differs')
    write(path, protocol)
    write(root/'manifest.json', manifest)
    if not (root/'checkpoint.pt').exists():
        try:
            os.link(checkpoint, root/'checkpoint.pt')
        except OSError:
            shutil.copyfile(checkpoint, root/'checkpoint.pt')
    if not (root/'model_config.yaml').exists():
        shutil.copyfile(config, root/'model_config.yaml')
    dataset = WindowDataset(root=str(source_root), split='val', split_file=str(split_file),
                            window_size=16, include_tx=False, video_ids=protocol['videos'], fixed_starts=[0], seed=SEED)
    assert len(dataset) == 12
    by_video = {m['video_id']: m for m in manifest}
    started = datetime.now(timezone.utc).isoformat()
    for count, dense in enumerate(DataLoader(dataset, batch_size=1, num_workers=0), 1):
        entry = by_video[dense['video_id'][0]]
        vi = entry['video_index']
        sparse, initial = materialize(dense, video_index=vi, rate=2, sigma=0., seed=SEED)
        noise = torch.stack([torch.randn(sparse['target'].shape[2:], generator=generator(vi, t, 1, seed=SEED))
                             for t in range(16)])[None]
        data = {k: sparse[k] for k in ('target', 'building', 'vehicle', 'valid_mask', 'sampling_mask')}
        data.update(measurement_noise=noise, initial=initial, video_id=entry['video_id'], start=0, video_index=vi)
        destination = root/entry['input_file']
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            old = torch.load(destination, weights_only=True)
            if old.keys() != data.keys() or any(not torch.equal(old[k], v) if torch.is_tensor(v) else old[k] != v for k, v in data.items()):
                raise ValueError(f'Existing input differs: {destination}')
        else:
            temporary = destination.with_suffix('.tmp')
            torch.save(data, temporary)
            temporary.replace(destination)
        state = dict(state='preparing', windows=count, total_windows=12, started_at=started)
        write(root/'prepare_status.json', state)
        print(json.dumps(state), flush=True)
    archive_path = root/'inputs.tar'
    if not archive_path.exists():
        temporary = archive_path.with_suffix('.tar.tmp')
        with tarfile.open(temporary, 'w') as archive:
            for name in ('checkpoint.pt', 'model_config.yaml', 'protocol.json', 'manifest.json', 'inputs'):
                archive.add(root/name, arcname=name)
        temporary.replace(archive_path)
    state.update(state='complete', calibration_videos=4, evaluation_videos=8,
                 archive=str(archive_path), archive_bytes=archive_path.stat().st_size,
                 finished_at=datetime.now(timezone.utc).isoformat())
    write(root/'prepare_status.json', state)
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=MAIN/'runs/noise_adaptive_remote_20261011')
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == '__main__':
    main()
