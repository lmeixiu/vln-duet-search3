"""Exercise the real rollout/graph/replay with a tiny CPU policy and fake simulator.

No MatterSim assets, pretrained weights, transformers or CUDA are required.
Only the model construction import and Tensor.cuda are replaced in this test.
"""
import copy
from collections import defaultdict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'map_nav_src'))
model_stub = ModuleType('models.model')
model_stub.VLNBert = object
model_stub.Critic = object
previous_model_module = sys.modules.get('models.model')
sys.modules['models.model'] = model_stub
try:
    from r2r.agent import GMapNavAgent
finally:
    if previous_model_module is None:
        sys.modules.pop('models.model', None)
    else:
        sys.modules['models.model'] = previous_model_module


class TinyPolicy(torch.nn.Module):
    def __init__(self, stop_bias=-100.):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0., 0., 0., 0., stop_bias]))

    def forward(self, mode, inputs):
        if mode == 'language':
            return torch.zeros(*inputs['txt_ids'].shape, 4) + self.bias[0] * 0
        if mode == 'panorama':
            embeds = inputs['view_img_fts'] + self.bias[0] * 0
            return embeds, torch.ones(embeds.shape[:2], dtype=torch.bool)
        rows = []
        for vpids in inputs['gmap_vpids']:
            idxs = [4 if vp is None else 'ABCD'.index(vp) for vp in vpids]
            values = self.bias[idxs]
            padding = inputs['gmap_masks'].shape[1] - len(vpids)
            rows.append(torch.cat([values, values.new_full((padding,), -float('inf'))]))
        logits = torch.stack(rows).masked_fill(inputs['gmap_visited_masks'], -float('inf'))
        return {'fused_logits': logits, 'global_logits': logits, 'local_logits': logits}


class FakeSim:
    def __init__(self, env, index):
        self.env, self.index = env, index

    def newEpisode(self, scan, vp, heading, elevation):
        self.env.positions[self.index] = vp[0]


class FakeEnv:
    # Edges are four meters long; stopping at A is a failure for goal B.
    links = {'A': ['B'], 'B': ['A', 'C'], 'C': ['B', 'D'], 'D': ['C']}

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.batch = [{'instr_id': str(i), 'path': ['A', 'B']} for i in range(batch_size)]
        self.env = SimpleNamespace(sims=[FakeSim(self, i) for i in range(batch_size)])
        self.positions = ['A'] * batch_size

    def reset(self, batch=None):
        if batch is not None:
            self.batch = list(batch)
        self.positions = [item['path'][0] for item in self.batch]
        return self._get_obs()

    def _get_obs(self):
        result = []
        for vp, item in zip(self.positions, self.batch):
            candidates = [{'viewpointId': dest, 'pointId': 12,
                           'feature': np.ones(8, dtype=np.float32),
                           'position': (0., float('ABCD'.index(dest) * 4), 0.)}
                          for dest in self.links[vp]]
            result.append({'instr_id': item['instr_id'], 'scan': 'scan', 'viewpoint': vp,
                           'viewIndex': 12, 'heading': 0., 'elevation': 0.,
                           'position': (0., float('ABCD'.index(vp) * 4), 0.),
                           'instr_encoding': [1, 2], 'candidate': candidates,
                           'feature': np.ones((36, 8), dtype=np.float32),
                           'gt_path': item['path']})
        return result


def make_agent(stop_bias=-100., horizon=5, batch_size=1):
    agent = GMapNavAgent.__new__(GMapNavAgent)
    agent.args = SimpleNamespace(
        max_action_len=horizon, fusion='dynamic', enc_full_graph=True,
        act_visited_nodes=False, image_feat_size=4, ignoreid=-100,
        detailed_output=False, policy_stop=False, replay_weight=0.2, batch_size=batch_size,
    )
    agent.env = FakeEnv(batch_size)
    agent.vln_bert = TinyPolicy(stop_bias)
    agent.scanvp_cands = {}
    agent.logs = defaultdict(list)
    agent.feedback = 'sample_policy'
    return agent


class PolicyRolloutTests(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor)
        self.cuda_patch.start()

    def tearDown(self):
        self.cuda_patch.stop()

    def collect(self, agent):
        with torch.no_grad():
            return agent.rollout(record_actions=True)

    def test_passes_ground_truth_goal_and_no_final_backtrack(self):
        agent = make_agent()
        record = self.collect(agent)[0]
        self.assertEqual(record['path'], [['A'], ['B'], ['C'], ['D']])
        self.assertEqual(record['steps'][-1]['forced'], 'no_candidates')
        self.assertEqual(record['steps'][-1]['log_prob'], 0.)
        self.assertTrue(all(step['action'] is not None for step in record['steps'][:-1]))

    def test_policy_can_stop_before_goal(self):
        record = self.collect(make_agent(stop_bias=100.))[0]
        self.assertEqual(record['path'], [['A']])
        self.assertIsNone(record['steps'][0]['action'])
        self.assertIsNone(record['steps'][0]['forced'])

    def test_exhausted_frontier_without_full_graph(self):
        agent = make_agent()
        agent.args.enc_full_graph = False
        record = self.collect(agent)[0]
        self.assertEqual(record['steps'][-1]['forced'], 'no_candidates')
        self.assertEqual(record['path'][-1], ['D'])

    def test_horizon_does_not_record_unexecuted_action(self):
        record = self.collect(make_agent(horizon=2))[0]
        self.assertEqual(record['path'], [['A'], ['B']])
        self.assertIsNone(record['steps'][-1]['action'])
        self.assertEqual(record['steps'][-1]['forced'], 'horizon')

    def test_replay_uses_current_logits_and_masks_forced_stop_loss(self):
        agent = make_agent(horizon=3)
        records = self.collect(agent)
        records[0]['loss_weight'] = 1.7
        with torch.no_grad():
            agent.vln_bert.bias.fill_(0.)  # New student: each valid choice has p=1/2.
        agent.feedback = 'replay'
        result = agent.rollout(replay=records)
        self.assertEqual(result[0]['path'], records[0]['path'])
        self.assertAlmostEqual(agent.replay_loss.item(), 1.7 * 2 * np.log(2), places=5)
        agent.replay_loss.backward()
        self.assertTrue(torch.isfinite(agent.vln_bert.bias.grad).all())
        self.assertGreater(agent.vln_bert.bias.grad[4].item(), 0.)

    def test_learned_stop_contributes_gradient(self):
        agent = make_agent(stop_bias=100.)
        records = self.collect(agent)
        records[0]['loss_weight'] = 1.
        with torch.no_grad():
            agent.vln_bert.bias.fill_(0.)
        agent.feedback = 'replay'
        agent.rollout(replay=records)
        agent.replay_loss.backward()
        self.assertLess(agent.vln_bert.bias.grad[4].item(), 0.)

    def test_diverged_observation_is_rejected(self):
        agent = make_agent(horizon=3)
        records = self.collect(agent)
        records[0]['loss_weight'] = 1.
        records[0]['steps'][1]['viewpoint'] = 'D'
        agent.feedback = 'replay'
        with self.assertRaisesRegex(ValueError, 'observation diverged'):
            agent.rollout(replay=records)

    def test_ended_batch_rows_have_no_extra_loss(self):
        agent = make_agent(horizon=3, batch_size=2)
        records = self.collect(agent)
        records[0]['path'] = [['A']]
        records[0]['steps'] = [dict(records[0]['steps'][0], action=None, forced=None)]
        for record in records:
            record['loss_weight'] = 1.
        with torch.no_grad():
            agent.vln_bert.bias.fill_(0.)
        agent.feedback = 'replay'
        agent.rollout(replay=records)
        self.assertAlmostEqual(agent.replay_loss.item(), 1.5 * np.log(2), places=5)
        agent.replay_loss.backward()
        self.assertTrue(torch.isfinite(agent.vln_bert.bias.grad).all())

    def test_legacy_argmax_keeps_backtrack_and_opt_in_uses_policy_stop(self):
        agent = make_agent(horizon=3)
        agent.feedback = 'argmax'
        legacy = agent.rollout()[0]
        self.assertEqual(legacy['path'][-1], ['B', 'A'])
        agent.args.policy_stop = True
        autonomous = agent.rollout()[0]
        self.assertEqual(autonomous['path'], [['A'], ['B'], ['C']])

    def test_optimizer_step_combines_original_il_and_replay(self):
        agent = make_agent(horizon=3)
        records = self.collect(agent)
        records[0]['loss_weight'] = 1.
        batch = copy.deepcopy(agent.env.batch)
        agent.replay_buffer = SimpleNamespace(sample=lambda size: (batch, records))
        agent.args.dataset = 'r2r'
        agent.args.train_alg = 'imitation'
        agent.args.aug = None
        agent.critic = torch.nn.Linear(1, 1)
        agent.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        agent.vln_bert_optimizer = torch.optim.SGD(agent.vln_bert.parameters(), lr=0.1)
        agent.critic_optimizer = torch.optim.SGD(agent.critic.parameters(), lr=0.1)
        before = agent.vln_bert.bias.detach().clone()
        agent.train(1)
        self.assertEqual(len(agent.logs['IL_loss']), 1)
        self.assertEqual(len(agent.logs['replay_loss']), 1)
        self.assertFalse(torch.equal(before, agent.vln_bert.bias))
        self.assertTrue(torch.isfinite(agent.vln_bert.bias).all())


if __name__ == '__main__':
    unittest.main()
