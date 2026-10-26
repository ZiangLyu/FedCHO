
import random
import sys

import numpy as np

from DDPG import DDPG
from DDPG import DDPG
from DDPG.DDPG import ExperienceReplayBuffer
import torch
from collections import defaultdict


class Server(object):
    def __init__(self, args):
        self.agent = DDPG.DDPG(args=args)
        self.aggre_workers = [[] for _ in range(args.aggre_num)]
        self.aggre_num = args.aggre_num

        state_dim = 13 * args.aggre_num  
        self.replay_buffer = ExperienceReplayBuffer(state_dim=state_dim, action_dim=self.aggre_num, max_size=10000)
        self.participation_count = {}
        self.ema_scores = {}
        self.malicious_counts = {}
        self.blacklist = set()
        self.KickR = 3
        self.performance_stats = {
              'task_completion_times': [],
              'device_utilization': {},
              'resource_efficiency': [],
              'capability_match_scores': []
          }

    def compute_state(self, device_list):
        state_list = []


        worker_devices = [d for d in device_list if d.return_role() == 'worker']

        for device in worker_devices:

            local_model = device.return_local_model()
            if device.is_online() and local_model is not None:
                local_accuracy = device.validate_model_weights(local_model.state_dict())
            else:
                local_accuracy = 0.0
            stake = device.return_stake()
            comp_power = device.return_computation_power()
            is_online = 1.0 if device.is_online() else 0.0


            data_size = len(device.train_dl.dataset) if hasattr(device, 'train_dl') else 0
            data_size_normalized = data_size / 5000.0


            ema_score = self.ema_scores.get(device.return_idx(), 0.0)
            participation_count = self.participation_count.get(device.return_idx(), 0)
            participation_normalized = min(participation_count / 10.0, 1.0)


            is_malicious = 1.0 if device.return_idx() in device.return_black_list() else 0.0
            malicious_count = self.malicious_counts.get(device.return_idx(), 0)
            malicious_normalized = min(malicious_count / self.KickR, 1.0)


            role_dict = {"worker": [1, 0, 0], "validator": [0, 1, 0], "miner": [0, 0, 1]}
            role_encoding = role_dict.get(device.return_role(), [0, 0, 0])


            link_speed_normalized = device.return_link_speed() / 70000.0 if hasattr(device, 'return_link_speed') else 1.0

            device_state = [
                local_accuracy, stake, comp_power, is_online, data_size_normalized,
                ema_score, participation_normalized, is_malicious, malicious_normalized,
                link_speed_normalized
            ] + role_encoding

            state_list.append(device_state)

        return torch.tensor(state_list, dtype=torch.float32)

    def topk_with_exploration(self, score_dict, k, epsilon=0.3):
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_m = [i for i, _ in sorted_scores[:max(2 * k, len(score_dict))]]

        num_explore = int(k * epsilon)
        num_exploit = k - num_explore

        top_k = top_m[:k]
        rest = list(set(top_m) - set(top_k))
        random.shuffle(rest)

        selected = top_k[:num_exploit] + rest[:num_explore]
        random.shuffle(selected)
        return selected[:k]

    def client_selection(self, devices_list, workers_needed, validators_needed, miners_needed):
        epsilon = 1e-6
        decay_rate = 0.05
        participation_decay = 1.0 - decay_rate


        current_state = self.compute_state(devices_list)


        worker_scores = {}
        validator_scores = {}
        miner_scores = {}

        for i, device in enumerate(devices_list):
            idx = device.return_idx()


            if idx in self.blacklist:
                continue

            local_model = device.return_local_model()
            if local_model is not None:
                acc = device.validate_model_weights(local_model.state_dict())
            else:
                acc = 0.0
            data_size = len(device.train_dl.dataset) if hasattr(device, 'train_dl') else 0
            time = device.local_update_time if device.local_update_time is not None else 1
            stake = device.return_stake()
            comp_power = device.return_computation_power()
            is_malicious = idx in device.return_black_list()

            if is_malicious:
                self.malicious_counts[idx] = self.malicious_counts.get(idx, 0) + 1
                if self.malicious_counts[idx] >= self.KickR:
                    self.blacklist.add(idx)
                continue
            else:
                self.malicious_counts[idx] = 0


            participation = self.participation_count.get(idx, 0)
            participation_penalty = participation * decay_rate


            worker_score = (0.4 * acc + 
                           0.25 * min(data_size/1000, 1.0) + 
                           0.2 * (1 / (time + epsilon)) + 
                           0.1 * comp_power + 
                           0.05 * stake) - participation_penalty
            worker_scores[idx] = worker_score


            validator_score = (0.5 * acc + 
                             0.3 * comp_power + 
                             0.2 * stake) - participation_penalty
            validator_scores[idx] = validator_score


            miner_score = (0.6 * stake + 
                          0.4 * comp_power) - participation_penalty
            miner_scores[idx] = miner_score


            avg_score = (worker_score + validator_score + miner_score) / 3
            self.ema_scores[idx] = 0.3 * avg_score + 0.7 * self.ema_scores.get(idx, 0)


        available = set(worker_scores.keys())
        selected_workers = self.topk_with_exploration(
            {k: v for k, v in worker_scores.items() if k in available}, 
            workers_needed, epsilon=0.2
        )
        available -= set(selected_workers)
        selected_validators = self.topk_with_exploration(
            {k: v for k, v in validator_scores.items() if k in available}, 
            validators_needed, epsilon=0.2
        )
        available -= set(selected_validators)
        selected_miners = self.topk_with_exploration(
            {k: v for k, v in miner_scores.items() if k in available}, 
            miners_needed, epsilon=0.2
        )


        all_selected = set(selected_workers + selected_validators + selected_miners)
        for idx in all_selected:
            self.participation_count[idx] = self.participation_count.get(idx, 0) + 1


        for device in devices_list:
            idx = device.return_idx()
            if idx not in all_selected and idx in self.participation_count:
                self.participation_count[idx] *= participation_decay


        workers_this_round, validators_this_round, miners_this_round = [], [], []

        for device in devices_list:
            idx = device.return_idx()
            if idx in selected_workers:
                device.assign_worker_role()
                workers_this_round.append(device)
            elif idx in selected_validators:
                device.assign_validator_role()
                validators_this_round.append(device)
            elif idx in selected_miners:
                device.assign_miner_role()
                miners_this_round.append(device)
            else:

                device.assign_role()
                role = device.return_role()
                if role == 'worker' and len(workers_this_round) < workers_needed:
                    workers_this_round.append(device)
                elif role == 'validator' and len(validators_this_round) < validators_needed:
                    validators_this_round.append(device)
                elif role == 'miner' and len(miners_this_round) < miners_needed:
                    miners_this_round.append(device)

            device.online_switcher()


        if len(workers_this_round) < workers_needed:
            print(f"Warning: Only {len(workers_this_round)} workers selected, needed {workers_needed}")
        if len(validators_this_round) < validators_needed:
            print(f"Warning: Only {len(validators_this_round)} validators selected, needed {validators_needed}")
        if len(miners_this_round) < miners_needed:
            print(f"Warning: Only {len(miners_this_round)} miners selected, needed {miners_needed}")


        next_state = self.compute_state(devices_list)

        next_worker_idx = [d.return_idx() for d in workers_this_round]


        return next_state.flatten(), workers_this_round, validators_this_round, miners_this_round,next_worker_idx

    def calculate_resource_efficiency(self, devices_list, selected_devices_indices):

        if not devices_list:
            return 0.0


        comp_powers = [d.return_computation_power() for d in devices_list]
        data_sizes = [len(d.train_dl.dataset) if hasattr(d, 'train_dl') else 0 for d in devices_list]
        stakes = [d.return_stake() for d in devices_list]


        max_comp_power = max(comp_powers) if comp_powers else 1
        max_data_size = max(data_sizes) if data_sizes else 1
        max_stake = max(stakes) if stakes else 1

        max_comp_power = max(max_comp_power, 1.0)
        max_data_size = max(max_data_size, 1.0)
        max_stake = max(max_stake, 1.0)


        POTENTIAL_WEIGHTS = {
            'comp': 0.4,
            'data': 0.3,
            'stake': 0.3
        }


        CONTRIBUTION_WEIGHTS = {
            'worker':    {'data': 0.5, 'comp': 0.4, 'stake': 0.1},
            'validator': {'comp': 0.7, 'stake': 0.3, 'data': 0.0},
            'miner':     {'stake': 0.6, 'comp': 0.4, 'data': 0.0}
        }

        total_potential = 0.0
        total_effective_contribution = 0.0

        for device in devices_list:

            norm_comp = device.return_computation_power() / max_comp_power
            norm_data = (len(device.train_dl.dataset) if hasattr(device, 'train_dl') else 0) / max_data_size
            norm_stake = device.return_stake() / max_stake


            device_potential = (norm_comp * POTENTIAL_WEIGHTS['comp'] +
                                norm_data * POTENTIAL_WEIGHTS['data'] +
                                norm_stake * POTENTIAL_WEIGHTS['stake'])
            total_potential += device_potential


            if device.return_idx() in selected_devices_indices:
                role = device.return_role()

                weights = CONTRIBUTION_WEIGHTS.get(role, {})

                effective_contribution = (norm_comp * weights.get('comp', 0) +
                                        norm_data * weights.get('data', 0) +
                                        norm_stake * weights.get('stake', 0))


                total_effective_contribution += min(effective_contribution, device_potential)


        if total_potential == 0:
            efficiency = 0.0
        else:
            efficiency = total_effective_contribution / total_potential


        self.performance_stats['resource_efficiency'].append(efficiency)
        print(f"本轮资源效率: {efficiency:.4f}")

        return efficiency