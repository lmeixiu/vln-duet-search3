"""Versioned offline trajectory data and instruction-balanced weighted replay.

This module deliberately has no torch/MatterSim dependency. Rewards are computed
only by the collector, after an episode, and are never passed to the policy.
"""
import hashlib
import json
import math
import random

FORMAT_VERSION = 1
POLICY = 'categorical_t1_policy_stop_no_backtrack_v1'
CONFIG_KEYS = (
    'dataset', 'fusion', 'enc_full_graph', 'graph_sprels', 'act_visited_nodes',
    'max_action_len', 'max_instr_len', 'tokenizer', 'features', 'image_feat_size',
    'angle_feat_size', 'obj_feat_size', 'num_l_layers', 'num_x_layers', 'num_pano_layers',
)


def policy_config(args):
    return {key: getattr(args, key) for key in CONFIG_KEYS}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def instruction_signature(item):
    # Includes the teacher path only to detect changes in the source annotations.
    fields = {key: item[key] for key in (
        'instr_id', 'scan', 'path', 'heading', 'instruction', 'instr_encoding')}
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def reward_from_metrics(metrics, success_weight=1., ndtw_weight=1., spl_weight=1.):
    coefficients = (success_weight, ndtw_weight, spl_weight)
    values = [float(metrics[key]) for key in ('success', 'nDTW', 'spl')]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Expected finite reward metrics on the [0, 1] scale')
    if any(not math.isfinite(v) or v < 0 for v in coefficients) or not any(coefficients):
        raise ValueError('Reward coefficients must be finite, nonnegative and not all zero')
    return sum(w * v for w, v in zip(coefficients, values))


def reward_weights(rewards, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('Replay beta must be finite and positive')
    if not rewards or any(not math.isfinite(r) for r in rewards):
        raise ValueError('Rewards must be nonempty and finite')
    maximum = max(rewards)
    values = [math.exp((r - maximum) / beta) for r in rewards]
    total = sum(values)
    return [v / total for v in values]


def validate_trajectory(record, instr_id, horizon, start_vp):
    if record.get('instr_id') != instr_id:
        raise ValueError('Trajectory instruction mismatch')
    steps = record.get('steps', [])
    if not 1 <= len(steps) <= horizon:
        raise ValueError('Invalid trajectory length')
    if steps[0]['viewpoint'] != start_vp or record['path'][0] != [start_vp]:
        raise ValueError('Trajectory start mismatch')
    if steps[-1]['action'] is not None:
        raise ValueError('Trajectory must end with a recorded STOP')
    for t, step in enumerate(steps):
        forced = step['forced']
        if forced not in (None, 'horizon', 'no_candidates'):
            raise ValueError('Unknown forced STOP reason')
        if step['action'] is None and t != len(steps) - 1:
            raise ValueError('Trajectory continues after STOP')
        if forced is not None and (step['action'] is not None or step['log_prob'] != 0):
            raise ValueError('Forced transitions must be STOP with log probability zero')
        if forced == 'horizon' and t != horizon - 1:
            raise ValueError('Premature horizon STOP')
        if not math.isfinite(step['log_prob']) or step['log_prob'] > 1e-6:
            raise ValueError('Invalid behavior log probability')
    # Each macro action appends one physical path segment; STOP appends none.
    if len(record['path']) != len(steps):
        raise ValueError('Physical path and action sequence disagree')
    for t, step in enumerate(steps[:-1]):
        segment = record['path'][t + 1]
        if not segment or segment[-1] != step['action'] or steps[t + 1]['viewpoint'] != step['action']:
            raise ValueError('Action destination and physical path disagree')


class ReplayBuffer:
    def __init__(self, path, args, instruction_data):
        self.rng = random.Random(args.seed + 971)
        self.groups = []
        annotations = {item['instr_id']: item for item in instruction_data}
        seen = set()
        with open(path, encoding='utf-8') as handle:
            meta = json.loads(next(handle))
            if meta.get('type') != 'metadata' or meta.get('version') != FORMAT_VERSION:
                raise ValueError('Unsupported replay format')
            if meta.get('split') != 'train' or meta.get('policy') != POLICY:
                raise ValueError('Replay requires train-only, untempered autonomous samples')
            if meta.get('config') != policy_config(args):
                raise ValueError('Replay/model configuration mismatch; match collection flags')
            sample_count = meta.get('samples_per_instruction', 0)
            if not isinstance(sample_count, int) or sample_count < 2:
                raise ValueError('Expected at least two samples per instruction')
            coefficients = meta['reward_weights']
            for line in handle:
                group = json.loads(line)
                instr_id = group['instr_id']
                if instr_id in seen or instr_id not in annotations:
                    raise ValueError('Duplicate or non-training replay instruction: %s' % instr_id)
                seen.add(instr_id)
                item = annotations[instr_id]
                if group['instruction_sha256'] != instruction_signature(item):
                    raise ValueError('Replay source annotations changed: %s' % instr_id)
                records = group['trajectories']
                if len(records) != sample_count:
                    raise ValueError('Incomplete candidate group')
                rewards = []
                for record in records:
                    validate_trajectory(record, instr_id, args.max_action_len, item['path'][0])
                    value = reward_from_metrics(record['metrics'], *coefficients)
                    if not math.isclose(value, record['reward'], rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError('Stored reward does not match metrics')
                    rewards.append(value)
                weights = reward_weights(rewards, args.replay_beta)
                for record, weight in zip(records, weights):
                    # Sample a candidate uniformly; K*w makes the expectation
                    # exactly sum_i w_i * trajectory_NLL_i for each instruction.
                    record['loss_weight'] = sample_count * weight
                self.groups.append((item, records))
        if not self.groups:
            raise ValueError('Replay file contains no trajectories')
        if len(self.groups) != meta.get('num_instructions'):
            raise ValueError('Replay file is truncated or has an incorrect instruction count')
        self.metadata = meta

    def sample(self, batch_size):
        batch, records = [], []
        for _ in range(batch_size):
            item, candidates = self.rng.choice(self.groups)
            batch.append(item)
            records.append(self.rng.choice(candidates))
        return batch, records
