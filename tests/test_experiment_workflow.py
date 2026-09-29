import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

from test_replay import fixture
from r2r.inspect_replay import inspect
from r2r.parser import parse_args

ROOT = Path(__file__).resolve().parents[1]


class ExperimentWorkflowTests(unittest.TestCase):
    def test_dry_run_all_variants_and_modes(self):
        bash = os.environ.get('BASH_FOR_TESTS') or shutil.which('bash')
        if not bash:
            self.skipTest('bash is not available')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'map_nav_src/scripts/run_r2r_experiment.sh'
            script.parent.mkdir(parents=True)
            with script.open('w', newline='\n') as handle:
                handle.write((ROOT / 'map_nav_src/scripts/run_r2r_experiment.sh').read_text())
            config = root / 'experiments/variant.env'
            config.parent.mkdir()
            env = dict(os.environ, DRY_RUN='1', BASE_CKPT='/data/base', DATA_ROOT='/data/datasets',
                       OUT_ROOT='/runs', RUN_ID='test', SEED='2', COLLECT_SEED='0',
                       TRAJECTORIES='/data/shared.jsonl')
            for variant in ('dagger', 'uniform', 'reward'):
                config.write_text('EXPERIMENT_VARIANT=%s\n' % variant)
                for mode in ('collect', 'train', 'eval-base', 'eval-best', 'eval-last', 'diagnose'):
                    with self.subTest(variant=variant, mode=mode):
                        result = subprocess.run([bash, script.as_posix(), mode], env=env,
                                                capture_output=True, text=True, check=True)
                        command = shlex.split(result.stdout.strip().splitlines()[-1])
                        self.assertEqual(command[:2], ['python', '-u'])
                        self.assertNotIn('--submit', command)
                        if mode == 'train':
                            self.assertEqual('--replay_file' in command, variant != 'dagger')
                            if variant != 'dagger':
                                self.assertEqual(command[command.index('--replay_mode') + 1], variant)
                            self.assertEqual(command[command.index('--iters') + 1], '5000')
                        if mode.startswith('eval'):
                            self.assertIn('--save_predictions', command)
                            self.assertEqual(command[-3:], ['val_train_seen', 'val_seen', 'val_unseen'])
                        if mode == 'eval-last':
                            self.assertTrue(command[command.index('--resume_file') + 1].endswith('latest_dict'))
            self.assertFalse((root / 'outputs').exists())

    def test_inspector_diversity_concentration_and_stop_counts(self):
        args, item, meta, group = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.jsonl'
            path.write_text(json.dumps(meta) + '\n' + json.dumps(group) + '\n')
            report = inspect(path, 1.)
            self.assertEqual(report['unique_action_fraction'], 0.5)
            self.assertEqual(report['reward_range'], 3.)
            self.assertGreater(report['max_weight'], 0.9)
            self.assertEqual(report['termination_counts']['policy'], 2)

    def test_validation_saving_does_not_enable_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(['--output_dir', tmp, '--test', '--save_predictions',
                               '--eval_splits', 'val_seen', 'val_unseen'])
            self.assertTrue(args.save_predictions)
            self.assertFalse(args.submit)
            self.assertEqual(args.eval_splits, ['val_seen', 'val_unseen'])


if __name__ == '__main__':
    unittest.main()
