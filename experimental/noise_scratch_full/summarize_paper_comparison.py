"""Paired comparison to the original paper model; keep tune/confirm separate."""
import argparse
import statistics
from pathlib import Path
import json
from experimental.noise_loss_screen.common import write

SIGMAS = [0, .03, .05, .09]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    root = Path(parser.parse_args().root)
    folder = root / 'paper_model_comparison'
    documents = {'paper_w16_step16000': json.loads((folder / 'matrix.json').read_text())}
    for name in ['best_reconstruction', 'best_alignment']:
        documents[name] = json.loads((root / 'training' / (name + '_matrix.json')).read_text())
    rows = {name: [r for r in d['rows'] if r['sigma'] in SIGMAS] for name, d in documents.items()}
    key = lambda r: (r['video_id'], r['start'], r['rate'], r['sigma'], r['input_sigma'])
    reference = {key(r): r for r in rows['paper_w16_step16000']}
    for name, rr in rows.items():
        assert len(rr) == len(reference) and {key(r) for r in rr} == set(reference)
        for r in rr:
            old = reference[key(r)]
            assert (r['base'], r['file'], r['group']) == (old['base'], old['file'], old['group'])
    selected = {}
    table = []
    for name, rr in rows.items():
        tune = [r for r in rr if r['group'] == 'tune']
        selected[name] = min(SIGMAS, key=lambda s: statistics.mean(r['metrics']['unobserved_mse'] for r in tune if r['input_sigma'] == s))
        for group in ['tune', 'confirm']:
            for rate in [None, 1, 2, 3]:
                for sigma in [None, *SIGMAS]:
                    part = [r for r in rr if r['group'] == group and (rate is None or r['rate'] == rate) and (sigma is None or r['sigma'] == sigma)]
                    for mode, fixed in [('correct', None), ('fixed_zero', 0), ('fixed_003', .03), ('tune_selected_fixed', selected[name])]:
                        chosen = [r for r in part if r['input_sigma'] == (r['sigma'] if mode == 'correct' else fixed)]
                        table.append(dict(model=name, group=group, rate=rate, sigma=sigma, mode=mode, fixed_sigma=fixed, cases=len(chosen),
                                          **{k: statistics.mean(r['metrics'][k] for r in chosen) for k in chosen[0]['metrics']}))
    result = dict(scope='Same fixed video windows, masks, observation noise, diffusion initial noise, FP32 DDIM50; validation only, no formal test',
                  selected_fixed_sigmas=selected, rows=table)
    write(folder / 'summary.json', result)
    for row in table:
        if row['group'] == 'confirm' and row['rate'] is None and row['sigma'] is None:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
