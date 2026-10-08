"""Validation-only stopping, retaining progress in either research objective."""
import math


class EarlyStop:
    def __init__(self, config, state=None):
        self.config = dict(config)
        self.state = dict(state or dict(score=None, gain=None, bad_checks=0, last_step=None, stop=False))

    def update(self, step, summaries):
        if self.state['last_step'] is not None and step <= self.state['last_step']:
            raise ValueError('Validation steps must increase; do not count a resumed check twice')
        scores = [float(s['correct_mse']) for s in summaries]
        gains = [float(s['alignment_gain']) for s in summaries]
        if not scores or not all(math.isfinite(x) for x in scores + gains):
            raise ValueError('Early stopping requires finite validation metrics')
        score, gain = min(scores), max(gains)
        old = self.state
        better_score = old['score'] is None or score < old['score'] * (1 - self.config['mse_relative_delta'])
        better_gain = old['gain'] is None or gain > old['gain'] + self.config['alignment_absolute_delta']
        if better_score:
            old['score'] = score
        if better_gain:
            old['gain'] = gain
        # Patience starts after the protected initial learning period.
        old['bad_checks'] = 0 if step <= self.config['min_steps'] or better_score or better_gain else old['bad_checks'] + 1
        old.update(last_step=step, stop=step > self.config['min_steps'] and old['bad_checks'] >= self.config['patience'],
                   improved_reconstruction=better_score, improved_alignment=better_gain)
        return old['stop']
