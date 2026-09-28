"""Collector -> JSONL -> replay integration, using the CPU simulation fixture."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from test_policy_rollout import FakeEnv, make_agent
from r2r.collect_trajectories import collect
from r2r.parser import parse_args
from r2r.replay import ReplayBuffer


class CollectionTests(unittest.TestCase):
    def test_nondivisible_batch_roundtrip_and_validation_guard(self):
        data = [{'instr_id': str(i), 'scan': 'scan', 'path': ['A', 'B'],
                 'heading': 0., 'instruction': 'go', 'instr_encoding': [1, 2]} for i in range(3)]

        def env_factory(*args, **kwargs):
            env = FakeEnv(kwargs['batch_size'])
            env._eval_item = lambda scan, path, gt: {'success': 0., 'nDTW': 0.5, 'spl': 0.}
            return env

        def agent_factory(args, env):
            agent = make_agent(horizon=args.max_action_len, batch_size=args.batch_size)
            agent.args = args
            agent.env = env
            agent.critic = torch.nn.Linear(1, 1)
            agent.models = (agent.vln_bert, agent.critic)
            agent.load = lambda path: 0
            return agent

        data_stub = ModuleType('utils.data')
        data_stub.ImageFeaturesDB = lambda *args: None
        env_stub = ModuleType('r2r.env')
        env_stub.R2RNavBatch = env_factory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoint'
            checkpoint.write_bytes(b'test checkpoint provenance')
            args = parse_args([
                '--output_dir', str(root / 'out'), '--trajectory_file', str(root / 'train.jsonl'),
                '--resume_file', str(checkpoint), '--fusion', 'dynamic', '--enc_full_graph',
                '--batch_size', '2', '--num_samples', '2', '--max_action_len', '3',
                '--image_feat_size', '4',
            ])
            with patch.dict(sys.modules, {'utils.data': data_stub, 'r2r.env': env_stub}), \
                 patch('r2r.data_utils.construct_instrs', return_value=copy.deepcopy(data)), \
                 patch('r2r.agent.GMapNavAgent', side_effect=agent_factory), \
                 patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
                collect(args)
                with self.assertRaises(FileExistsError):
                    collect(args)
            output = Path(args.trajectory_file)
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(lines), 4)  # metadata + 3 real instructions, no padding
            self.assertEqual({line['instr_id'] for line in lines[1:]}, {'0', '1', '2'})
            self.assertTrue(all(len(line['trajectories']) == 2 for line in lines[1:]))
            self.assertFalse(Path(str(output) + '.partial').exists())
            summary = json.loads(Path(str(output) + '.summary.json').read_text())
            self.assertEqual(summary['num_instructions'], 3)
            self.assertAlmostEqual(summary['mean_reward'], 0.5)
            buffer = ReplayBuffer(output, args, data)
            self.assertEqual(len(buffer.groups), 3)
            # The actual collected actions can be replayed by a fresh student.
            batch, records = buffer.sample(2)
            student = make_agent(horizon=3, batch_size=2)
            student.env.reset(batch=batch)
            student.feedback = 'replay'
            with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
                student.rollout(reset=False, replay=records)
                student.replay_loss.backward()
            self.assertTrue(torch.isfinite(student.vln_bert.bias.grad).all())
            lines[0]['split'] = 'val_unseen'
            output.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
            with self.assertRaisesRegex(ValueError, 'train-only'):
                ReplayBuffer(output, args, data)


if __name__ == '__main__':
    unittest.main()
