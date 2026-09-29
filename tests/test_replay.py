import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'map_nav_src'))
from r2r.replay import (
    CONFIG_KEYS, FORMAT_VERSION, POLICY, ReplayBuffer, instruction_signature,
    policy_config, reward_from_metrics, reward_weights, validate_trajectory,
)


def fixture():
    args = SimpleNamespace(**{key: False for key in CONFIG_KEYS})
    args.dataset = 'r2r'
    args.max_action_len = 3
    args.seed = 0
    args.replay_beta = 1.
    item = {'instr_id': '1_0', 'scan': 'scan', 'path': ['a', 'b'], 'heading': 0.,
            'instruction': 'go', 'instr_encoding': [1, 2]}
    record = {'instr_id': '1_0', 'path': [['a']], 'steps': [
        {'viewpoint': 'a', 'view_index': 12, 'action': None, 'forced': None, 'log_prob': -1.}],
        'metrics': {'success': 0., 'nDTW': 0., 'spl': 0.}, 'reward': 0.}
    meta = {'type': 'metadata', 'version': FORMAT_VERSION, 'split': 'train', 'policy': POLICY,
            'config': policy_config(args), 'samples_per_instruction': 2,
            'num_instructions': 1, 'reward_weights': [1., 1., 1.]}
    good = copy.deepcopy(record)
    good['metrics'] = {'success': 1., 'nDTW': 1., 'spl': 1.}
    good['reward'] = 3.
    group = {'instr_id': '1_0', 'instruction_sha256': instruction_signature(item),
             'trajectories': [record, good]}
    return args, item, meta, group


class ReplayTests(unittest.TestCase):
    def load_buffer(self, args, item, meta, group):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'replay.jsonl'
            path.write_text(json.dumps(meta) + '\n' + json.dumps(group) + '\n', encoding='utf-8')
            return ReplayBuffer(path, args, [item])

    def test_stable_weights_and_unbiased_candidate_scaling(self):
        self.assertEqual(reward_weights([10000., 10000.], 0.01), [0.5, 0.5])
        weights = reward_weights([0., 3.], 1.)
        self.assertAlmostEqual(weights[1], math.exp(3) / (1 + math.exp(3)))
        args, item, meta, group = fixture()
        buffer = self.load_buffer(args, item, meta, group)
        records = buffer.groups[0][1]
        losses = [7., 2.]
        actual = sum(x['loss_weight'] * loss for x, loss in zip(records, losses)) / 2
        self.assertAlmostEqual(actual, sum(w * loss for w, loss in zip(weights, losses)))
        batch, sampled = buffer.sample(8)
        self.assertEqual(len(batch), 8)
        self.assertTrue(all(x['instr_id'] == item['instr_id'] for x in sampled))

    def test_invalid_weights_and_reward_scale(self):
        for beta in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                reward_weights([0., 1.], beta)
        with self.assertRaises(ValueError):
            reward_from_metrics({'success': 100., 'nDTW': 1., 'spl': 1.})

    def test_uniform_replay_ignores_reward_for_loss_weights(self):
        args, item, meta, group = fixture()
        args.replay_mode = 'uniform'
        buffer = self.load_buffer(args, item, meta, group)
        self.assertEqual([r['loss_weight'] for r in buffer.groups[0][1]], [1., 1.])
        args.replay_mode = 'reward'
        reward_buffer = self.load_buffer(args, item, meta, group)
        self.assertGreater(reward_buffer.groups[0][1][1]['loss_weight'], 1.)

    def test_reject_validation_incomplete_and_changed_config(self):
        for mutation in ('validation', 'incomplete', 'config', 'annotation', 'reward', 'policy'):
            args, item, meta, group = fixture()
            if mutation == 'validation':
                meta['split'] = 'val_unseen'
            elif mutation == 'incomplete':
                group['trajectories'].pop()
            elif mutation == 'config':
                meta['config']['fusion'] = 'local'
            elif mutation == 'annotation':
                item['heading'] = 1.
            elif mutation == 'reward':
                group['trajectories'][0]['reward'] = 8.
            else:
                meta['policy'] = 'temperature_0.5'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load_buffer(args, item, meta, group)

    def test_reject_invalid_termination(self):
        _, _, _, group = fixture()
        record = group['trajectories'][0]
        record['steps'][0]['forced'] = 'horizon'
        record['steps'][0]['log_prob'] = 0.
        with self.assertRaises(ValueError):
            validate_trajectory(record, '1_0', 3, 'a')


if __name__ == '__main__':
    unittest.main()
