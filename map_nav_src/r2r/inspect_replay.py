"""CPU-only inspection of candidate diversity and reward concentration."""
import argparse
import json
import math
from pathlib import Path

from r2r.replay import reward_weights


def inspect(path, beta):
    totals = dict(unique_action_fraction=0., reward_range=0., max_weight=0.,
                  effective_sample_fraction=0., mean_success=0., oracle_success_at_k=0.)
    stops = {'policy': 0, 'horizon': 0, 'no_candidates': 0}
    groups = 0
    equal_reward_groups = 0
    with open(path, encoding='utf-8') as handle:
        metadata = json.loads(next(handle))
        for line in handle:
            group = json.loads(line)
            records = group['trajectories']
            k = len(records)
            weights = reward_weights([r['reward'] for r in records], beta)
            actions = {tuple(s['action'] for s in r['steps']) for r in records}
            spread = max(r['reward'] for r in records) - min(r['reward'] for r in records)
            totals['unique_action_fraction'] += len(actions) / k
            totals['reward_range'] += spread
            totals['max_weight'] += max(weights)
            totals['effective_sample_fraction'] += 1. / (k * sum(w*w for w in weights))
            totals['mean_success'] += sum(r['metrics']['success'] for r in records) / k
            totals['oracle_success_at_k'] += max(r['metrics']['success'] for r in records)
            equal_reward_groups += math.isclose(spread, 0., abs_tol=1e-8)
            for record in records:
                stops[record['steps'][-1]['forced'] or 'policy'] += 1
            groups += 1
    if not groups or groups != metadata['num_instructions']:
        raise ValueError('Empty or incomplete collection')
    result = {key: value/groups for key, value in totals.items()}
    result.update(groups=groups, split=metadata['split'], beta=beta,
                  equal_reward_group_fraction=equal_reward_groups/groups,
                  termination_counts=stops,
                  termination_fractions={k: v/sum(stops.values()) for k, v in stops.items()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory_file')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(inspect(args.trajectory_file, args.beta), indent=2)
    print(report)
    if args.output:
        Path(args.output).write_text(report + '\n', encoding='utf-8')
