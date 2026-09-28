"""Collect K independent, autonomous DUET episodes per instruction.

Run from map_nav_src with python -m r2r.collect_trajectories. Validation
collections are diagnostic only and are rejected by the replay trainer.
"""
import json
import os
from pathlib import Path

from r2r.parser import parse_args
from r2r.replay import (
    FORMAT_VERSION, POLICY, file_sha256, instruction_signature, policy_config,
    reward_from_metrics, validate_trajectory,
)


def collect(args):
    if args.dataset != 'r2r' or args.world_size != 1 or args.aug is not None:
        raise ValueError('Collector v1 supports single-GPU R2R without --aug')
    if not args.resume_file or not args.trajectory_file:
        raise ValueError('--resume_file and --trajectory_file are required')
    if args.num_samples < 2 or args.batch_size < 1 or args.max_action_len < 2 or args.collect_limit < 0:
        raise ValueError('Require K >= 2, batch_size >= 1, horizon >= 2 and limit >= 0')
    output = Path(args.trajectory_file)
    partial = output.with_name(output.name + '.partial')
    summary_path = output.with_name(output.name + '.summary.json')
    if output.exists() or partial.exists() or summary_path.exists():
        raise FileExistsError('Use a new trajectory_file; existing collection will not be overwritten')
    coefficients = [args.reward_success, args.reward_ndtw, args.reward_spl]
    reward_from_metrics({'success': 0., 'nDTW': 0., 'spl': 0.}, *coefficients)

    # Heavy dependencies are delayed so --help works without MatterSim/CUDA.
    import torch
    from utils.misc import set_random_seed
    from utils.data import ImageFeaturesDB
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    from r2r.agent import GMapNavAgent

    set_random_seed(args.seed)
    data = construct_instrs(args.anno_dir, args.dataset, [args.collect_split],
                            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len)
    data.sort(key=lambda item: item['instr_id'])
    if args.collect_limit:
        data = data[:args.collect_limit]
    if not data:
        raise ValueError('No collection instructions found')
    instructions = list(data)  # env constructor shuffles its own list
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    env = R2RNavBatch(feat_db, list(data), args.connectivity_dir,
                      batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,
                      seed=args.seed, name=args.collect_split)
    agent = GMapNavAgent(args, env)
    agent.load(args.resume_file)
    agent.vln_bert.eval()
    agent.critic.eval()
    agent.feedback = 'sample_policy'
    for model in agent.models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    metadata = {
        'type': 'metadata', 'version': FORMAT_VERSION, 'split': args.collect_split,
        'policy': POLICY, 'config': policy_config(args), 'seed': args.seed,
        'samples_per_instruction': args.num_samples, 'num_instructions': len(instructions),
        'checkpoint': os.path.abspath(args.resume_file),
        'checkpoint_sha256': file_sha256(args.resume_file),
        'reward_weights': coefficients,
    }
    totals = {'mean_success': 0., 'oracle_success_at_k': 0., 'mean_reward': 0.,
              'best_reward': 0., 'mean_spl': 0., 'mean_ndtw': 0.}
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open('x', encoding='utf-8') as handle, torch.no_grad():
        handle.write(json.dumps(metadata) + '\n')
        for offset in range(0, len(instructions), args.batch_size):
            real_batch = instructions[offset:offset + args.batch_size]
            # Pad only simulator slots; never write padded duplicates.
            batch = real_batch + [real_batch[-1]] * (args.batch_size - len(real_batch))
            groups = [{'instr_id': item['instr_id'],
                       'instruction_sha256': instruction_signature(item),
                       'trajectories': []} for item in real_batch]
            for sample_id in range(args.num_samples):
                env.reset(batch=batch)
                trajectories = agent.rollout(reset=False, record_actions=True)
                for i, group in enumerate(groups):
                    record = trajectories[i]
                    record.pop('details', None)
                    record['sample_id'] = sample_id
                    # Ground truth enters only here, after the whole rollout.
                    metrics = env._eval_item(batch[i]['scan'], record['path'], batch[i]['path'])
                    record['metrics'] = {key: float(metrics[key]) for key in ('success', 'nDTW', 'spl')}
                    record['reward'] = reward_from_metrics(record['metrics'], *coefficients)
                    validate_trajectory(record, group['instr_id'], args.max_action_len, batch[i]['path'][0])
                    group['trajectories'].append(record)
            for group in groups:
                candidates = group['trajectories']
                k = len(candidates)
                totals['mean_success'] += sum(x['metrics']['success'] for x in candidates) / k
                totals['oracle_success_at_k'] += max(x['metrics']['success'] for x in candidates)
                totals['mean_reward'] += sum(x['reward'] for x in candidates) / k
                totals['best_reward'] += max(x['reward'] for x in candidates)
                totals['mean_spl'] += sum(x['metrics']['spl'] for x in candidates) / k
                totals['mean_ndtw'] += sum(x['metrics']['nDTW'] for x in candidates) / k
                handle.write(json.dumps(group, allow_nan=False) + '\n')
            handle.flush()
            print('Collected %d/%d instructions (K=%d)' % (
                min(offset + args.batch_size, len(instructions)), len(instructions), args.num_samples), flush=True)
    os.replace(str(partial), str(output))
    summary = {key: value / len(instructions) for key, value in totals.items()}
    summary.update({'num_instructions': len(instructions), 'samples_per_instruction': args.num_samples,
                    'split': args.collect_split, 'oracle_is_diagnostic_only': True})
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    collect(parse_args())
