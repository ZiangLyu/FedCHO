import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
from DatasetLoad import AddGaussianNoise
from torch import optim
import random
import copy
import time
from sys import getsizeof

from Crypto.PublicKey import RSA
from hashlib import sha256
from Models import Mnist_2NN, Mnist_CNN
from Blockchain import Blockchain
import os
import math
from phe.encoding import EncodedNumber
import phe.paillier

from phe import paillier

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DatasetManager.DatasetManager import split_dataset, split_dataset_dirichlet_safe
import matplotlib.pyplot as plt


from DatasetManager.DatasetManager import FederatedDatasetManager
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from collections import defaultdict


def generate_public_dataset(num_samples=1000, image_size=(3, 32, 32)):

    data = torch.randn(num_samples, *image_size)

    data = (data - data.min()) / (data.max() - data.min())
    return DataLoader(data, batch_size=64, shuffle=True)


class Device:
    def __init__(self, idx, attack_type,assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti,
                 network_stability, net, dev, miner_acception_wait_time, miner_accepted_transactions_size_limit,
                 validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed,
                 even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain,
                 malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds):
        self.idx = idx

        self.attack_type = attack_type
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.loss_KL_func = torch.nn.KLDivLoss(reduction='batchmean').to(dev)
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)
        if opti == "SGD":
            self.opti = optim.SGD(self.net.parameters(), lr=learning_rate)
        else:
            self.opti = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.dev = dev

        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True, drop_last=True)
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None

        self.role = None
        self.pow_difficulty = pow_difficulty
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.devices_dict = None
        self.aio = False
        if even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.peer_list = set()

        self.online = True
        self.rewards = 0
        self.blockchain = Blockchain()

        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.private_key, self.public_key = paillier.generate_paillier_keypair()

        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount

        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}


        self.round_end_time = 0
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.validator_associated_worker_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None

        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = validator_threshold
        self.validator_local_accuracy = None
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()

        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit

        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        self.unordered_propagated_block_processing_queue = {}
        self.variance_of_noises = None or []
        self.generate_rsa_key()
        self.global_model = None
        self.local_model = None
        self.public_dataset = generate_public_dataset(num_samples=1000)
        self.validation_stats_records = []


    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e

    def init_global_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
        self.global_parameters = self.net.state_dict()
        self.global_model = copy.deepcopy(self.net)

    def assign_role(self):

        role_choice = random.randint(0, 2)
        if role_choice == 0:
            self.role = "worker"
        elif role_choice == 1:
            self.role = "miner"
        else:
            self.role = "validator"


    def assign_miner_role(self):
        self.role = "miner"

    def assign_worker_role(self):
        self.role = "worker"

    def assign_validator_role(self):
        self.role = "validator"


    def return_idx(self):
        return self.idx

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

    def return_is_malicious(self):
        return self.is_malicious

    def return_black_list(self):
        return self.black_list

    def return_blockchain_object(self):
        return self.blockchain

    def return_stake(self):
        return self.rewards

    def return_validation_stats_records(self):
        return self.validation_stats_records

    def clear_validation_stats_records(self):
        self.validation_stats_records = []

    def return_computation_power(self):
        return self.computation_power

    def return_the_added_block(self):
        return self.the_added_block

    def return_round_end_time(self):
        return self.round_end_time


    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')

        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def online_switcher(self):
        old_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True

            if old_status == False:
                print(f"{self.idx} goes back online.")

                self.update_peer_list()

                if self.pow_resync_chain():
                    self.update_model_after_chain_resync()
        else:
            self.online = False
            print(f"{self.idx} goes offline.")
        return self.online

    def update_peer_list(self):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)

        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())

        self.remove_peers(self)

        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)

        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                for peer in removed_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()

        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):

        pow_proof = block_to_check.return_pow_proof()


        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[
                    chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True

    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.devices_dict[block.return_mined_by()].return_stake()
        return accumulated_stake

    def resync_chain(self, mining_consensus):
        if self.not_resync_chain:
            return
        if mining_consensus == 'PoW':
            self.pow_resync_chain()
        else:
            self.pos_resync_chain()

    def pos_resync_chain(self):
        print(
            f"{self.role} {self.idx} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A chain from {peer.return_idx()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")

                        curr_chain_stake = peer_chain_stake
                        highest_stake_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(
                            f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:

            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()

            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")

            return True
        print("Chain not resynced.")
        return False

    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")

                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(
                            f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:

            longest_chain_structure = longest_chain.return_chain_structure()

            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")

            return True
        print("Chain not resynced.")
        return False

    def receive_rewards(self, rewards):
        self.rewards += rewards

    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.black_list:
            print(f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            signature = transaction_to_verify["miner_signature"]

            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                  byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True

    def verify_block(self, block_to_verify, sending_miner):
        if not self.online_switcher():
            print(f"{self.idx} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            print(
                f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        if mined_by in self.black_list:
            print(f"The miner {mined_by} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False, False

        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.idx} is not verified.")
            return False, False

        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()
            modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()

            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(),
                                  byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                print(
                    f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}.")
                return False, False

            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:

                last_block_hash = last_block.compute_hash(hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(
                        f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device needs to resync chain next round.")
                    return False, False

        print(f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.idx}!")
        if self.computation_power == 0:
            verification_time = float('inf')
        else:
            verification_time = (time.time() - verification_time) / self.computation_power
        return block_to_verify, verification_time

    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(
            f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")


        self.the_added_block = block_to_add
        return True


    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, device_idx_to_action,
                      client_models, local_epochs,
                      when_resync=False):

        processing_time = time.time()
        if not self.online_switcher():
            print(
                f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.")
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:

                print(
                    f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:


                self_rewards_accumulator = 0
                valid_transactions_records_by_worker = {}
                valid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()[
                    'valid_validator_sig_transacitons']
                comm_round = block_to_process.return_block_idx()
                self.active_worker_record_by_round[comm_round] = set()
                for valid_validator_sig_worker_transaciton in valid_validator_sig_worker_transacitons_in_block:

                    if self.verify_miner_transaction_by_signature(valid_validator_sig_worker_transaciton, mined_by):
                        worker_device_idx = valid_validator_sig_worker_transaciton['worker_device_idx']
                        self.active_worker_record_by_round[comm_round].add(worker_device_idx)
                        if not worker_device_idx in valid_transactions_records_by_worker.keys():
                            valid_transactions_records_by_worker[worker_device_idx] = {}
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = None

                        local_epoch_seq = valid_validator_sig_worker_transaciton[
                            'local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_worker_transaciton[
                            'positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_worker_transaciton[
                            'negative_direction_validators']
                        if len(positive_direction_validators) >= len(negative_direction_validators):

                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'].add(
                                local_epoch_seq)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(
                                local_epoch_seq)

                            if local_epoch_seq == max(
                                    valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = \
                                    valid_validator_sig_worker_transaciton['local_updates_params']

                            if self.idx == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaciton[
                                    'local_updates_rewards']
                        else:
                            if self.malicious_updates_discount:

                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(
                                    local_epoch_seq)
                                valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(
                                    local_epoch_seq)

                                if local_epoch_seq == max(
                                        valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):

                                    discounted_valid_validator_sig_worker_transaciton_local_updates_params = copy.deepcopy(
                                        valid_validator_sig_worker_transaciton['local_updates_params'])
                                    for var in discounted_valid_validator_sig_worker_transaciton_local_updates_params:
                                        discounted_valid_validator_sig_worker_transaciton_local_updates_params[
                                            var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_worker[worker_device_idx][
                                        'finally_used_params'] = discounted_valid_validator_sig_worker_transaciton_local_updates_params

                                if self.idx == worker_device_idx:
                                    self_rewards_accumulator += valid_validator_sig_worker_transaciton[
                                                                    'local_updates_rewards'] * self.malicious_updates_discount
                            else:

                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(
                                    local_epoch_seq)


                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.idx == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.idx == validator_record['miner_device_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return


                self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
                for worker_idx, local_updates_direction_records in valid_transactions_records_by_worker.items():
                    if len(local_updates_direction_records['negative_epochs']) > len(
                            local_updates_direction_records['positive_epochs']):
                        self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                        kick_out_accumulator = 1

                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round.keys():
                                if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:

                            self.black_list.add(worker_idx)

                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.devices_dict[worker_idx].return_is_malicious():
                                msg = f"{self.idx} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)",
                                                    (worker_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f"WARNING: {self.idx} has mistakenly regard {worker_idx} as a malicious worker device in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)",
                                                    (worker_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)


                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()[
                    'invalid_validator_sig_transacitons']
                for invalid_validator_sig_worker_transaciton in invalid_validator_sig_worker_transacitons_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_worker_transaciton, mined_by):
                        validator_device_idx = invalid_validator_sig_worker_transaciton['validator']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(validator_device_idx)
                        kick_out_accumulator = 1

                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round.keys():
                                if validator_device_idx in self.untrustworthy_validators_record_by_comm_round[
                                    comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:

                            self.black_list.add(validator_device_idx)
                            print(
                                f"{validator_device_idx} has been regarded as a compromised validator by {self.idx} in {comm_round}.")


                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return

                    if self.idx == invalid_validator_sig_worker_transaciton['miner_device_idx']:
                        self_rewards_accumulator += invalid_validator_sig_worker_transaciton[
                            'miner_rewards_for_this_tx']

                if self.idx == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()

                self.receive_rewards(self_rewards_accumulator)
                print(
                    f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")

                finally_used_local_params = []

                TP, FP, TN, FN = 0, 0, 0, 0
                for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                    is_worker_malicious = self.devices_dict[worker_device_idx].return_is_malicious()
                    if local_params_record['finally_used_params']:

                        finally_used_local_params.append(
                            (worker_device_idx, local_params_record['finally_used_params']))
                        if not is_worker_malicious:
                            TP += 1
                        else:
                            FP += 1
                    else:

                        if is_worker_malicious:
                            TN += 1
                        else:
                            FN += 1
                if self.online_switcher():
                    self.global_update(finally_used_local_params,device_idx_to_action)


                else:
                    print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")

        malicious_worker_validation_log_path = f"{log_files_folder_path}/comm_{comm_round}/malicious_worker_validation_log.txt"
        if not os.path.exists(malicious_worker_validation_log_path):
            with open(malicious_worker_validation_log_path, 'w') as file:
                accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN) else 0
                precision = TP / (TP + FP) if TP else 0
                recall = TP / (TP + FN) if TP else 0
                f1 = precision * recall / (precision + recall) if precision * recall else 0
                file.write(f"In comm_{comm_round} of validating workers, TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}. \
                        \nAccuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
        if self.computation_power == 0:
            processing_time = float('inf')
        else:
            processing_time = (time.time() - processing_time) / self.computation_power
        return processing_time

    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add

    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_idx() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1

                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1,
                                                         this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_idx() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:

                            self.black_list.add(device.return_idx())
                            msg = f"worker {device.return_idx()} has been regarded as a lazy worker by {self.idx} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'a') as file:
                                file.write(msg)
                else:

                    pass

    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):

        self.global_parameters = copy.deepcopy(self.initial_net_parameters)

        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path, conn, conn_cursor, when_resync=True)

    def return_pow_difficulty(self):
        return self.pow_difficulty

    def register_in_the_network(self, check_online=False):
        if self.aio:
            self.add_peers(set(self.devices_dict.values()))
        else:
            potential_registrars = set(self.devices_dict.values())

            potential_registrars.discard(self)

            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:
                        return False
                    registrar = random.sample(online_registrars, 1)[0]

            self.add_peers(registrar)

            self.add_peers(registrar.return_peers())

            registrar.add_peers(self)
            return True


    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                print("{} add noise".format(self.idx))
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))

    def malicious_worker_sign_flipping_to_weights(self,m):
        with torch.no_grad():
            if hasattr(m, 'weight'):

                m.weight.mul_(-1)

    def malicious_worker_same_value_to_weights(self,m):
        with torch.no_grad():
            if hasattr(m, 'weight'):

                attack_value = 10.0
                m.weight.fill_(attack_value)


    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, eta_c, gamma_c,
                            local_epochs=1):
        print(
            f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed, 3)} bytes/s")

        self.local_model.load_state_dict(self.global_model.state_dict(), strict=True)
        self.local_update_time = time.time()


        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0
        self.local_model.train()
        local_epoch_norm_max = 0.0
        local_norm_sum = 0.0


        for epoch in range(local_epochs):
            running_loss = 0.0
            running_corrects = 0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.local_model(data)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_func(outputs, label)
                self.local_model.zero_grad()
                loss.backward()
                state_dict = self.local_model.state_dict()
                for name, param in self.local_model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm(p=2)
                        step_size = min(eta_c, (gamma_c * eta_c) / grad_norm)
                        state_dict[name] -= step_size * param.grad
                self.local_model.load_state_dict(state_dict)
                local_epoch_norm = torch.sqrt(
                    sum(param.grad.norm(p=2) ** 2 for name, param in self.local_model.named_parameters()))
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == label.data)
                if local_epoch_norm >= local_epoch_norm_max:
                    local_epoch_norm_max = local_epoch_norm
                local_norm_sum += local_epoch_norm
                self.local_updates_rewards_per_transaction += rewards * (label.shape[0])


            with open(
                    f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt",
                    "a") as file:
                file.write(
                    f"{self.return_idx()} epoch_{epoch + 1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.return_local_model().state_dict())}\n")
            self.local_total_epoch += 1

        try:
            self.local_update_time = (time.time() - self.local_update_time) / self.computation_power
        except:
            self.local_update_time = float('inf')
        if self.is_malicious:
            if self.attack_type == "gaussian":
                print('🔥 attack type is gussian')
                self.local_model.apply(self.malicious_worker_add_noise_to_weights)
                print(f"❗️ malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            elif self.attack_type == "sign_flipping":
                print('🔥 attack type is sign_flipping')
                self.local_model.apply(self.malicious_worker_sign_flipping_to_weights)
                print(f"❗️ malicious worker {self.idx} has flipped the sign of its local updated weights before transmitting")
            elif self.attack_type == "same_value":
                print('🔥 attack type is same_value')
                self.local_model.apply(self.malicious_worker_same_value_to_weights)
                print(f"❗️ malicious worker {self.idx} has set all its local updated weights to the same value before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(
                    f"{self.return_idx()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")

        with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt",
                  "a") as file:
            file.write(
                f"{self.return_idx()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.return_local_model().state_dict())}\n")
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.return_local_model().state_dict()

        return self.local_update_time

    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']

            if opti == 'SGD':
                validation_opti = optim.SGD(validation_net.parameters(), lr=currently_used_lr)
            local_update_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = validation_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            return (time.time() - local_update_time) / self.computation_power, validation_net.state_dict()

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def return_local_updates_and_signature(self, comm_round):


        local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round,
                              "local_updates_params": self.local_train_parameters,
                              "local_updates_rewards": self.local_updates_rewards_per_transaction,
                              "local_iteration(s)_spent_time": self.local_update_time,
                              "local_total_accumulated_epochs_this_round": self.local_total_epoch,
                              "worker_rsa_pub_key": self.return_rsa_pub_key()}

        local_updates_dict["worker_signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        return local_updates_dict

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.variance_of_noises.clear()
        self.round_end_time = 0

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = received_block
            print(
                f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(
                f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")

    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner


    def validate_model_weights(self, weights_to_eval=None):
        self.temporal_model = copy.deepcopy(self.net)
        self.temporal_model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            if weights_to_eval:
                self.temporal_model.load_state_dict(weights_to_eval, strict=True)
            else:
                self.temporal_model.load_state_dict(self.global_parameters, strict=True)
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.temporal_model(data)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_func(outputs, label)

                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == label.data)

        val_loss = running_loss / len(self.test_dl.dataset)
        val_acc = running_corrects.double() / len(self.test_dl.dataset)

        del self.temporal_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        return val_acc.cpu().item()


    def global_update(self, local_update_params_potentially_to_be_used, rl_weights: dict):
        if not local_update_params_potentially_to_be_used:
            print(f"No local parameters to aggregate for {self.idx}")
            return

        valid_params = []
        valid_weights = []
        device_indices = []


        for (device_idx, local_params) in local_update_params_potentially_to_be_used:
            if device_idx in self.black_list:
                print(f"Skipping blacklisted device {device_idx}")
                continue


            rl_weight = max(rl_weights.get(device_idx, 0.1), 0.01)

            valid_params.append(local_params)
            valid_weights.append(rl_weight)
            device_indices.append(device_idx)

        if not valid_params:
            print(f"No valid parameters for global update in {self.idx}")
            return


        adaptive_weights = self._compute_adaptive_weights(valid_params, valid_weights, device_indices)


        aggregated_params = self._weighted_aggregation(valid_params, adaptive_weights)


        aggregated_params = self._apply_momentum_and_regularization(aggregated_params)


        self.global_parameters = aggregated_params
        self.global_model.load_state_dict(self.global_parameters)


    def _compute_adaptive_weights(self, local_params_list, rl_weights, device_indices):
        adaptive_weights = []


        similarities = []
        for local_params in local_params_list:
            similarity = self._compute_model_similarity(self.global_parameters, local_params)
            similarities.append(similarity)


        if max(similarities) > min(similarities):
            norm_similarities = [(s - min(similarities)) / (max(similarities) - min(similarities)) 
                               for s in similarities]
        else:
            norm_similarities = [1.0] * len(similarities)


        for i, (rl_w, sim) in enumerate(zip(rl_weights, norm_similarities)):

            sim_weight = 1.0 - abs(sim - 0.7)
            sim_weight = max(sim_weight, 0.1)


            adaptive_weight = 0.7 * rl_w + 0.3 * sim_weight
            adaptive_weights.append(adaptive_weight)


        total_weight = sum(adaptive_weights)
        if total_weight > 0:
            adaptive_weights = [w / total_weight for w in adaptive_weights]
        else:

            adaptive_weights = [1.0 / len(adaptive_weights)] * len(adaptive_weights)

        return adaptive_weights

    def _compute_model_similarity(self, params1, params2):
        vec1 = []
        vec2 = []

        for key in params1.keys():
            if key in params2:
                vec1.append(params1[key].flatten())
                vec2.append(params2[key].flatten())

        if not vec1:
            return 0.0

        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)


        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return cos_sim.item()

    def _weighted_aggregation(self, local_params_list, weights):
        aggregated_params = {}


        for key in local_params_list[0].keys():
            aggregated_params[key] = torch.zeros_like(local_params_list[0][key], dtype=torch.float32)


        for local_params, weight in zip(local_params_list, weights):
            for key in aggregated_params.keys():
                if key in local_params:
                    aggregated_params[key] += weight * local_params[key].float()

        return aggregated_params

    def _apply_momentum_and_regularization(self, aggregated_params):
        momentum = 0.9
        regularization_strength = 1e-5

        if hasattr(self, '_previous_global_params') and self._previous_global_params is not None:

            for key in aggregated_params.keys():
                if key in self._previous_global_params:
                    velocity = aggregated_params[key] - self._previous_global_params[key]
                    aggregated_params[key] = self._previous_global_params[key] + momentum * velocity


        self._previous_global_params = {k: v.clone() for k, v in aggregated_params.items()}


        for key in aggregated_params.keys():
            aggregated_params[key] *= (1 - regularization_strength)

        return aggregated_params


    def aggregate_models(self, local_update_params_potentially_to_be_used, device_idx_to_action, client_models,
                         local_epochs, bit,
                         quantize):

        total_weight = sum(device_idx_to_action.values())
        if total_weight != 0:
            device_idx_to_action = {k: v / total_weight for k, v in device_idx_to_action.items()}

        global_dict = copy.deepcopy(self.global_model.state_dict())
        param_diffs = {name: torch.zeros_like(param).float() for name, param in global_dict.items()}

        local_params_by_benign_workers = []
        benign_workers_idxs = []
        benign_workers_weights = []
        for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used:
            if not worker_device_idx in self.black_list:
                local_params_by_benign_workers.append(local_params)
                benign_workers_idxs.append(worker_device_idx)
            else:
                print(f"global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")

        if local_params_by_benign_workers:

            selected_models = {idx: model for idx, model in client_models.items() if idx in benign_workers_idxs}

            for idx, client_model in selected_models.items():
                client_dict = copy.deepcopy(client_model.state_dict())
                for name in global_dict.keys():
                    param_diffs[name] += device_idx_to_action[idx] * (global_dict[name] - client_dict[name])


            num_clients = len(benign_workers_idxs)
            eta_s = 0.05
            eta_c = 0.01
            gamma_s = 1
            num_epochs_per_round = local_epochs

            param_diffs_norm = torch.sqrt(
                sum(torch.norm(param_diffs[name] / (num_clients), p=2) ** 2 for name in param_diffs.keys()))

            global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm / (num_clients * num_epochs_per_round)))

            for name in global_dict.keys():
                global_dict[name] = global_dict[name].float() - global_step_size * (
                        param_diffs[name] / (num_clients * eta_c)).float()

            self.global_model.load_state_dict(global_dict)

            for var in global_dict:
                self.global_parameters[var] = global_dict[var]

        for var in self.global_parameters:
            self.global_parameters[var] = global_dict[var]


    def request_to_download(self, block_to_download, requesting_time_point):
        print(f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_validator_set.union(self.miner_associated_worker_set)
        for device in devices_in_association:

            if self.online_switcher() and device.online_switcher():
                miner_link_speed = self.return_link_speed()
                device_link_speed = device.return_link_speed()
                lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = getsizeof(str(block_to_download.__dict__)) / lower_link_speed
                verified_block, verification_time = device.verify_block(block_to_download,
                                                                        block_to_download.return_mined_by())
                if verified_block:

                    device.add_block(verified_block)
                device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(
                    f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")

    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        if peer.online_switcher():
                            peer.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")

    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__)) / lower_link_speed
            self.unordered_propagated_block_processing_queue[
                source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(
                f"{self.role} {self.idx} has accepted accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(
                f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")

    def add_propagated_block_to_processing_queue(self, arrival_time, propagated_block):
        self.unordered_propagated_block_processing_queue[arrival_time] = propagated_block

    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue

    def return_associated_validators(self):
        return self.miner_associated_validator_set

    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time

    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def verify_validator_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']
            if transaction_validator_idx in self.black_list:
                print(f"{transaction_validator_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed["validator_signature"]
                modulus = transaction_to_verify['validator_rsa_pub_key']["modulus"]
                pub_key = transaction_to_verify['validator_rsa_pub_key']["pub_key"]
                signature = transaction_to_verify["validator_signature"]

                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                      byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (time.time() - verification_time) / self.computation_power
                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from validator {transaction_validator_idx} is NOT verified.")
                    return (time.time() - verification_time) / self.computation_power, False
            else:
                print(
                    f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                verification_time = (time.time() - verification_time) / self.computation_power
                return verification_time, True

    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time) / self.computation_power
        return signing_time

    def mine_block(self, candidate_block, rewards, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        pow_mined_block = self.proof_of_work(candidate_block)

        pow_mined_block.set_mining_rewards(rewards)
        return pow_mined_block

    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()

        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()


        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    def return_block_generation_time_point(self):
        return self.block_generation_time_point

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = received_propagated_block
            print(
                f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(
                f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = received_propagated_validator_block
            print(
                f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(
                f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block

    def toss_propagated_block(self):
        self.received_propagated_block = None

    def toss_ropagated_validator_block(self):
        self.received_propagated_validator_block = None

    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()


        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        self.unordered_arrival_time_accepted_validator_transactions.clear()
        self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None

        self.unordered_propagated_block_processing_queue.clear()
        self.round_end_time = 0

    def set_unordered_arrival_time_accepted_validator_transactions(self,
                                                                   unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions

    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions

    def miner_broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(
                            self.unordered_arrival_time_accepted_validator_transactions)

                        for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del \
                                    final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[
                                        arrival_time]
                        peer.accept_miner_broadcasted_validator_transactions(self,
                                                                             final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                        print(
                            f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_validator_transactions(self, source_device,
                                                        unordered_transaction_arrival_queue_from_source_miner):

        if not source_device.return_idx() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append(
                {'source_device_link_speed': source_device.return_link_speed(),
                 'broadcasted_transactions': unordered_transaction_arrival_queue_from_source_miner})
            print(
                f"{self.role} {self.idx} has accepted validator transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(
                f"Source miner {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions

    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine


    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0

        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_worker_set.clear()


        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0

    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)

    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue

    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list

    def return_validations_and_signature(self, comm_round):
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_number': comm_round,
                                       'accuracies_this_round': copy.deepcopy(self.accuracies_this_round),
                                       'validation_effort_rewards': self.validation_rewards_this_round,
                                       "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(
                f"WARNING: {worker_device.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(miners_in_peer_list, 1)[0]
        return self.validator_associated_miner


    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
        else:
            print(
                f"WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_device_idx):
        if not souce_device_idx in self.black_list:
            self.unconfirmmed_transactions.append(unconfirmmed_transaction)
            print(f"{souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(
                f"Source device {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is broadcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(self, self.unconfirmmed_transactions)
                    else:
                        print(
                            f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. broadcasting skipped.")

    def accept_broadcasted_transactions(self, source_device, broadcasted_transactions):

        if not source_device.return_idx() in self.black_list:
            self.broadcasted_transactions.append(broadcasted_transactions)
            print(
                f"{self.role} {self.idx} has accepted transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(
                f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")


    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_device(self, to_associate_device_role):
        to_associate_device = vars(self)[f'{self.role}_associated_{to_associate_device_role}']
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:

            if peer.return_role() == to_associate_device_role and peer.is_online():
                if not peer.return_idx() in self.black_list:
                    to_associate_device = peer
        if not to_associate_device:

            return False
        print(
            f"{self.role} {self.idx} associated with {to_associate_device.return_role()} {to_associate_device.return_idx()}")
        return to_associate_device


    def set_unordered_arrival_time_accepted_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_transactions = unordered_transaction_arrival_queue

    def return_unordered_arrival_time_accepted_worker_transactions(self):
        return self.unordered_arrival_time_accepted_worker_transactions

    def validator_broadcast_worker_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "validator":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"validator {self.idx} is broadcasting received validator transactions to validator {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator = copy.copy(
                            self.unordered_arrival_time_accepted_worker_transactions)

                        for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del \
                                    final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator[
                                        arrival_time]

                        peer.accept_validator_broadcasted_worker_transactions(self,
                                                                              final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)
                        print(
                            f"validator {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)} worker transactions to validator {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination validator {peer.return_idx()} is in this validator {self.idx}'s black_list. broadcasting skipped for this dest validator.")

    def accept_validator_broadcasted_worker_transactions(self, source_validator,
                                                         unordered_transaction_arrival_queue_from_source_validator):
        if not source_validator.return_idx() in self.black_list:
            self.validator_accepted_broadcasted_worker_transactions.append(
                {'source_validator_link_speed': source_validator.return_link_speed(),
                 'broadcasted_transactions': unordered_transaction_arrival_queue_from_source_validator})
            print(
                f"validator {self.idx} has accepted worker transactions from validator {source_validator.return_idx()}")
        else:
            print(
                f"Source validator {source_validator.return_idx()} is in validator {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_worker_transactions(self):
        return self.validator_accepted_broadcasted_worker_transactions

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate

    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self, opti):

        print(f"validator {self.idx} is performing one epoch of local update and validation")
        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and will not be able to complete this validation")
            return float('inf')
        else:
            updated_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']

            if opti == 'SGD':
                validation_opti = optim.SGD(updated_net.parameters(), lr=currently_used_lr)
            else:
                validation_opti = optim.Adam(updated_net.parameters(), lr=currently_used_lr)
            local_validation_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = updated_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()


            corrects = 0
            with torch.no_grad():
                for data, label in self.test_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    outputs = updated_net(data)
                    _, preds = torch.max(outputs, 1)
                    corrects += torch.sum(preds == label.data)

            self.validator_local_accuracy = corrects.double() / len(self.test_dl.dataset)
            print(
                f"validator {self.idx} locally updated model has accuracy {self.validator_local_accuracy} on its local test set")
            return (time.time() - local_validation_time) / self.computation_power

    def return_local_updates_and_signature_ex(self, comm_round):
        local_updates_dict = {}
        local_updates_dict['ori'] = self.private_key.encrypt(
            {'worker_device_idx': self.idx, 'in_round_number': comm_round,
             "local_updates_params": copy.deepcopy(self.local_train_parameters),
             "local_updates_rewards": self.local_updates_rewards_per_transaction,
             "local_iteration(s)_spent_time": self.local_update_time,
             "local_total_accumulated_epochs_this_round": self.local_total_epoch,
             "worker_rsa_pub_key": self.return_rsa_pub_key()})
        local_updates_dict["worker_signature"] = self.sign_msg(sorted(local_updates_dict.items()))
        local_updates_dict['worker_device_idx'] = self.idx
        return local_updates_dict

    def validate_worker_transaction_ex(self, transaction_to_validate, rewards, log_files_folder_path, comm_round,
                                       malicious_validator_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"

        worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
        transaction_to_validate = self.devices_dict[worker_transaction_device_idx].public_key.decrypt(
            transaction_to_validate['ori'])

        if self.computation_power == 0:
            print(
                f"validator {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_validate)
                del transaction_before_signed["worker_signature"]
                modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['worker_rsa_pub_key']["pub_key"]
                signature = transaction_to_validate["worker_signature"]


                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                      byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                    transaction_to_validate['worker_signature_valid'] = True
                else:
                    print(
                        f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")

                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(
                    f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                transaction_to_validate['worker_signature_valid'] = True

            if transaction_to_validate['worker_signature_valid']:

                accuracy_by_worker_update_using_own_data = self.validate_model_weights_by_subset(
                    transaction_to_validate["local_updates_params"])


                print(f'validator updated model accuracy - {self.validator_local_accuracy}')
                print(
                    f"After applying worker's update, model accuracy becomes - {accuracy_by_worker_update_using_own_data}")

                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(
                        f"{log_files_folder_path_comm_round}/validator_{self.idx}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt",
                        "a") as file:
                    is_malicious_node = "M" if self.devices_dict[
                        worker_transaction_device_idx].return_is_malicious() else "B"
                    file.write(
                        f"{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}: validator {self.return_idx()} {is_malicious_validator} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")
                if accuracy_by_worker_update_using_own_data - self.validator_local_accuracy < self.validator_threshold * -1:
                    transaction_to_validate['update_direction'] = False
                    print(
                        f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by validator {self.idx}")

                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")

                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(
                        f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.idx}")

                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")

                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                if self.is_malicious and malicious_validator_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                        file.write(
                            f"malicious validator {self.idx} has flipped the voting of worker {worker_transaction_device_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['validation_rewards'] = rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time) / self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()

            transaction_to_validate["validator_signature"] = self.sign_msg(sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate

    def cosine_similarity(self, x, y, dim=0, eps=1e-8):
        dot_product = torch.sum(x * y, dim=dim)
        norm_x = torch.norm(x, dim=dim)
        norm_y = torch.norm(y, dim=dim)
        similarity = dot_product / (norm_x * norm_y + eps)
        return similarity

    def compute_layerwise_similarity(self, model_a, model_b, layers_to_check):
        if not layers_to_check:
            return 0.0


        similarities = []
        for layer_name in layers_to_check:
            if layer_name in model_a and layer_name in model_b:
                a = model_a[layer_name].flatten()
                b = model_b[layer_name].flatten()


                sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
                similarities.append(sim)

        if similarities:

            if len(similarities) > 3:
                weights = np.exp(np.linspace(0, 1, len(similarities)))
                weights = weights / weights.sum()
                return np.average(similarities, weights=weights)
            else:
                return sum(similarities) / len(similarities)
        else:
            return 0.0

    def __init_progressive_validation(self):
        if not hasattr(self, 'node_profiles'):
            self.node_profiles = {}
            self.validation_cache = {}
            self.phase_thresholds = [0.6, 0.7, 0.8]
            self.similarity_buffer = {}

    def _get_progressive_threshold(self, worker_idx, comm_round):

        if comm_round <= 5:
            base_threshold = 0.4
        elif comm_round <= 15:
            base_threshold = 0.5 + (comm_round - 5) * 0.02
        elif comm_round <= 30:
            base_threshold = 0.7
        else:
            base_threshold = 0.75


        if worker_idx not in self.node_profiles:
            self.node_profiles[worker_idx] = {
                'reputation_score': 0.5,
                'consistency_score': 0.5,
                'recent_similarities': [],
                'validation_count': 0,
                'success_history': []
            }

        profile = self.node_profiles[worker_idx]


        reputation = profile['reputation_score']
        consistency = profile['consistency_score'] 
        experience = min(1.0, profile['validation_count'] / 20)


        trust_score = 0.5 * reputation + 0.3 * consistency + 0.2 * experience


        trust_factor = 1.3 - 0.6 * trust_score


        adaptive_threshold = base_threshold * trust_factor


        adaptive_threshold = max(0.25, min(0.85, adaptive_threshold))

        return adaptive_threshold

    def _enhance_decision(self, base_decision, similarity, threshold, worker_idx, comm_round):

        if comm_round <= 10:
            return base_decision

        profile = self.node_profiles.get(worker_idx, {})
        trust_score = self._calculate_trust_score(profile)


        margin = abs(similarity - threshold)

        if margin < 0.05:
            if trust_score > 0.8 and not base_decision:

                return True
            elif trust_score < 0.2 and base_decision:

                return False

        return base_decision

    def _calculate_trust_score(self, profile):
        reputation = profile.get('reputation_score', 0.5)
        consistency = profile.get('consistency_score', 0.5)
        validation_count = profile.get('validation_count', 0)


        experience_weight = min(1.0, validation_count / 15)


        trust_score = 0.5 * reputation + 0.3 * consistency + 0.2 * experience_weight
        return trust_score

    def _update_worker_profile(self, worker_idx, similarity, decision, comm_round):
        if worker_idx not in self.node_profiles:
            return

        profile = self.node_profiles[worker_idx]
        profile['validation_count'] += 1


        if len(profile['recent_similarities']) >= 20:
            profile['recent_similarities'].pop(0)
        profile['recent_similarities'].append(similarity)


        if 'success_history' not in profile:
            profile['success_history'] = []
        if len(profile['success_history']) >= 30:
            profile['success_history'].pop(0)
        profile['success_history'].append(decision)


        alpha = 0.1
        if decision:
            profile['reputation_score'] = (1 - alpha) * profile['reputation_score'] + alpha * 1.0
        else:
            profile['reputation_score'] = (1 - alpha) * profile['reputation_score'] + alpha * 0.2


        if len(profile['recent_similarities']) >= 5:
            similarity_std = np.std(profile['recent_similarities'][-10:])
            consistency = 1.0 / (1.0 + similarity_std)
            if 'consistency_score' not in profile:
                profile['consistency_score'] = 0.5
            profile['consistency_score'] = (1 - alpha) * profile['consistency_score'] + alpha * consistency


        if comm_round % 5 == 0:

            cache_keys_to_remove = [k for k in self.validation_cache.keys() if str(worker_idx) in k]
            for key in cache_keys_to_remove:
                del self.validation_cache[key]

    def validate_worker_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round,
                                    malicious_validator_on):


        self.__init_progressive_validation()

        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"

        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and cannot validate this transaction.")
            return False, False

        worker_idx = transaction_to_validate['worker_device_idx']
        if worker_idx in self.black_list:
            print(f"{worker_idx} is in validator's blacklist. Transaction won't be validated.")
            return False, False

        validation_time_start = time.time()


        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_validate)
            del transaction_before_signed["worker_signature"]
            modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
            pub_key = transaction_to_validate['worker_rsa_pub_key']["pub_key"]
            signature = transaction_to_validate["worker_signature"]

            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                  byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)

            transaction_to_validate['worker_signature_valid'] = (hash == hashFromSignature)
            if hash == hashFromSignature:
                pass

            else:
                print(f"Invalid signature from worker {worker_idx}.")
        else:
            transaction_to_validate['worker_signature_valid'] = True


        if transaction_to_validate['worker_signature_valid']:
            validator_model_dict = self.local_model.state_dict()
            worker_model_dict = transaction_to_validate["local_updates_params"]


            layers_to_check = [k for k in validator_model_dict.keys() if
                               'weight' in k and ('c' in k or 'fc' in k or 'l_c' in k)]

            similarity = self.compute_layerwise_similarity(validator_model_dict, worker_model_dict, layers_to_check)


            adaptive_threshold = self._get_progressive_threshold(worker_idx, comm_round)


            base_decision = similarity >= adaptive_threshold


            if comm_round > 10:
                final_decision = self._enhance_decision(base_decision, similarity, adaptive_threshold, worker_idx, comm_round)
                base_decision = final_decision

            transaction_to_validate['update_direction'] = base_decision

            print(f"Progressive validation: similarity={similarity:.4f}, threshold={adaptive_threshold:.4f}, "
                  f"decision={base_decision} for worker {worker_idx}")


            self._update_worker_profile(worker_idx, similarity, base_decision, comm_round)


            worker_device = self.devices_dict.get(worker_idx)
            validation_record = {
                'comm_round': comm_round,
                'validator_id': self.idx,
                'worker_id': worker_idx,
                'is_malicious_worker': worker_device.is_malicious if worker_device else False,
                'similarity': similarity,
                'adaptive_threshold': adaptive_threshold,
                'validation_result': base_decision,
                'validation_time': time.time() - validation_time_start
            }
            self.validation_stats_records.append(validation_record)


            if self.is_malicious and malicious_validator_on:
                old_vote = transaction_to_validate['update_direction']
                transaction_to_validate['update_direction'] = not old_vote
                with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                    file.write(
                        f"Malicious validator {self.idx} flipped vote for worker {worker_idx} from {old_vote} to {transaction_to_validate['update_direction']} in round {comm_round}\n")

            transaction_to_validate['validation_rewards'] = rewards
        else:
            transaction_to_validate['update_direction'] = 'N/A'
            transaction_to_validate['validation_rewards'] = 0

        transaction_to_validate['validation_done_by'] = self.idx
        validation_time = (time.time() - validation_time_start) / self.computation_power
        transaction_to_validate['validation_time'] = validation_time
        transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()
        transaction_to_validate["validator_signature"] = self.sign_msg(sorted(transaction_to_validate.items()))

        return validation_time, transaction_to_validate

    def return_global_parameters(self):
        return self.global_parameters

    def return_global_model(self):
        return self.global_model

    def return_local_model(self):
        return self.local_model

    def set_local_model(self, model):
        self.local_model = model

    def set_local_parameters(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("Model parameters must be a dictionary")
        parameters_copy = parameters
        for var in parameters_copy:
            self.local_parameters[var] = parameters_copy[var]

    def return_local_parameters(self):
        return self.local_train_parameters

    def validate_model_weights_by_subset(self, weights_to_eval=None, subset_ratio=0.1):
        self.net.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)
            else:
                self.net.load_state_dict(self.global_parameters, strict=True)
            subset_size = int(len(self.test_dl.dataset) * subset_ratio)
            indices = random.sample(range(len(self.test_dl.dataset)), subset_size)
            subset = torch.utils.data.Subset(self.test_dl.dataset, indices)
            subset_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)
            for data, label in subset_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                outputs = self.net(data)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_func(outputs, label)
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == label.data)
            val_loss = running_loss / subset_size
            val_acc = running_corrects.double() / subset_size
            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        return val_acc


class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, alpha,attack_type,batch_size, learning_rate, loss_func, opti, num_devices,
                 network_stability, net, dev, knock_out_rounds, lazy_worker_knock_out_rounds, shard_test_data,
                 miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty,
                 even_link_speed_strength, base_data_transmission_speed, even_computation_power,
                 malicious_updates_discount, num_malicious, noise_variance, check_signature, not_resync_chain):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.alpha = alpha
        self.attack_type = attack_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = dict()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds

        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed_strength = even_link_speed_strength
        self.base_data_transmission_speed = base_data_transmission_speed
        self.even_computation_power = even_computation_power
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain

        self.validator_threshold = validator_threshold
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.pow_difficulty = pow_difficulty
        self.data_set_balanced_allocation()


    def data_set_balanced_allocation(self):
        print(f"\n🎯 Use Device change")
        print(f"\n🎯 Dataset: {self.data_set_name.lower()}")
        print(f"\n🎯 num malicious: {self.num_malicious}")
        print(f"\n🎯 dirichlet alpha: {self.alpha}")

        if self.data_set_name.lower() == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=transform_train)
            val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_val)
        elif self.data_set_name.lower() == 'mnist':
            train_transform = transforms.Compose([


                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])


            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=train_transform)
            val_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                         transform=test_transform)
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_val = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            data_dir = '/dataset/tiny-imagenet-200'
            train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
            val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_val)

        client_datasets = split_dataset_dirichlet_safe(train_dataset=train_dataset, num_clients=self.num_devices, alpha=self.alpha,
                                        min_samples_per_client=10)

        malicious_client_ids = random.sample(range(self.num_devices), self.num_malicious) if self.num_malicious else []

        for idx in malicious_client_ids:
            print(f"client {idx + 1} is malicious node")

        for i in range(self.num_devices):
            device_idx = f'device_{i + 1}'
            print(
                'client {} Train Dataset length is {} Test Dataset length is {}'.format(i + 1, len(client_datasets[i]),
                                                                                        len(DataLoader(
                                                                                            val_dataset,
                                                                                            batch_size=100,
                                                                                            shuffle=False,
                                                                                            drop_last=True).dataset)))
            a_device = Device(idx=device_idx,
                              attack_type = self.attack_type,
                              assigned_train_ds=client_datasets[i],
                              assigned_test_dl=DataLoader(val_dataset, batch_size=100, shuffle=False,
                                                          drop_last=True),
                              local_batch_size=self.batch_size,
                              learning_rate=self.learning_rate,
                              loss_func=self.loss_func,
                              opti=self.opti,
                              network_stability=self.default_network_stability,
                              net=self.net,
                              dev=self.dev,
                              miner_acception_wait_time=self.miner_acception_wait_time,
                              miner_accepted_transactions_size_limit=self.miner_accepted_transactions_size_limit,
                              validator_threshold=self.validator_threshold,
                              pow_difficulty=self.pow_difficulty,
                              even_link_speed_strength=self.even_link_speed_strength,
                              base_data_transmission_speed=self.base_data_transmission_speed,
                              even_computation_power=self.even_computation_power,
                              is_malicious=True if i in malicious_client_ids else False,
                              noise_variance=self.noise_variance,
                              check_signature=self.check_signature,
                              not_resync_chain=self.not_resync_chain,
                              malicious_updates_discount=self.malicious_updates_discount,
                              knock_out_rounds=self.knock_out_rounds,
                              lazy_worker_knock_out_rounds=self.lazy_worker_knock_out_rounds)

            self.devices_set[device_idx] = a_device

            print(f"Sharding dataset to {device_idx} done.")

        print("\n✅ Sharding dataset done!")


