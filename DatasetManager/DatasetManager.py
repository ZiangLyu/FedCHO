
import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FederatedDatasetManager(object):

    def __init__(self,
                 dataset_name: str = 'mnist',
                 num_clients: int = 10,
                 batch_size: int = 32,
                 is_iid: bool = True,
                 split_test: bool = False,
                 malicious_client_ids: Optional[List[int]] = None,
                 noise_level: float = 0.1,
                 seed: int = 42):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.is_iid = is_iid
        self.split_test = split_test
        self.malicious_client_ids = malicious_client_ids or []
        self.noise_level = noise_level
        self.seed = seed


        np.random.seed(seed)
        torch.manual_seed(seed)


        self.train_data, self.train_labels, self.test_data, self.test_labels = self._load_dataset()


        self.client_data = {}


        self._allocate_data()

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dataset_name.lower() == 'mnist':

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])


            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform)

        elif self.dataset_name.lower() == 'cifar10':

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])


            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_val)

        elif self.dataset_name.lower() == 'fashionmnist':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

            train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                              transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                             transform=transform)


        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")


        train_data = torch.stack([x for x, _ in train_dataset])
        train_labels = torch.tensor([y for _, y in train_dataset])

        test_data = torch.stack([x for x, _ in test_dataset])
        test_labels = torch.tensor([y for _, y in test_dataset])

        print(f"成功加载数据集 {self.dataset_name}")
        print(f"训练集大小: {train_data.shape}, 测试集大小: {test_data.shape}")

        return train_data, train_labels, test_data, test_labels

    def _allocate_data(self, dirichlet_alpha: float = 0.5):
        num_classes = len(torch.unique(self.train_labels))
        class_indices = [torch.where(self.train_labels == i)[0] for i in range(num_classes)]

        client_indices = [[] for _ in range(self.num_clients)]

        if self.is_iid:

            all_indices = torch.randperm(len(self.train_labels))
            splits = torch.chunk(all_indices, self.num_clients)
            for i in range(self.num_clients):
                client_indices[i] = splits[i]
        else:

            min_size = 0
            while min_size < 10:
                client_indices = [[] for _ in range(self.num_clients)]
                for c, idx_c in enumerate(class_indices):
                    idx_c = idx_c[torch.randperm(len(idx_c))]
                    proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, self.num_clients))
                    proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                    splits = np.split(idx_c.numpy(), proportions)
                    for i, split in enumerate(splits):
                        client_indices[i].extend(split)
                sizes = [len(idx) for idx in client_indices]
                min_size = min(sizes)


        for client_id in range(self.num_clients):
            indices = torch.tensor(client_indices[client_id])
            client_train_data = self.train_data[indices]
            client_train_labels = self.train_labels[indices]


            if client_id in self.malicious_client_ids:

                print(f"客户端 {client_id} 是恶意节点")


            if self.split_test:
                test_chunk_size = len(self.test_labels) // self.num_clients
                start_idx = client_id * test_chunk_size
                end_idx = (client_id + 1) * test_chunk_size if client_id != self.num_clients - 1 else len(
                    self.test_labels)
                client_test_data = self.test_data[start_idx:end_idx]
                client_test_labels = self.test_labels[start_idx:end_idx]
            else:
                client_test_data = self.test_data
                client_test_labels = self.test_labels


            train_loader = DataLoader(
                TensorDataset(client_train_data, client_train_labels),
                batch_size=self.batch_size,
                shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(client_test_data, client_test_labels),
                batch_size=100,
                shuffle=False
            )

            self.client_data[client_id] = {
                'train_loader': train_loader,
                'test_loader': test_loader,
                'num_train_samples': len(client_train_data),
                'num_test_samples': len(client_test_data),
                'is_malicious': client_id in self.malicious_client_ids
            }

            print(f"客户端 {client_id}: 训练样本 = {len(client_train_data)}, 测试样本 = {len(client_test_data)}")

    def _add_noise(self, data: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(data) * self.noise_level
        return data + noise

    def get_client_data(self, client_id: int) -> dict:
        if client_id not in self.client_data:
            raise ValueError(f"客户端ID {client_id} 不存在")
        return self.client_data[client_id]

    def get_non_iid_visualization(self) -> dict:
        label_distribution = {}

        for client_id, client_info in self.client_data.items():

            labels = torch.cat([y for _, y in client_info['train_loader'].dataset])


            unique_labels, counts = torch.unique(labels, return_counts=True)


            distribution = {int(label.item()): int(count.item()) for label, count in zip(unique_labels, counts)}

            label_distribution[client_id] = distribution

        return label_distribution


import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import random_split


def split_dataset(train_dataset, num_clients, alpha=0.5, iid=True, min_samples_per_client=10):

    if num_clients <= 0:
        raise ValueError("Number of clients must be positive")
    if min_samples_per_client * num_clients > len(train_dataset):
        raise ValueError("Not enough samples for requested clients and minimum samples per client")


    if hasattr(train_dataset, 'targets'):
        train_labels = train_dataset.targets
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.numpy()
        else:
            train_labels = np.array(train_labels)
    elif hasattr(train_dataset, 'samples'):
        train_labels = np.array([s[1] for s in train_dataset.samples])
    else:
        raise ValueError("Dataset format not supported. Could not find labels.")

    num_classes = np.max(train_labels) + 1
    total_samples = len(train_dataset)

    if iid:

        base_size = total_samples // num_clients
        lengths = [base_size] * num_clients
        remainder = total_samples % num_clients


        for i in range(remainder):
            lengths[i] += 1


        for i in range(num_clients):
            if lengths[i] < min_samples_per_client:

                excess = [j for j, l in enumerate(lengths) if l > min_samples_per_client]
                if not excess:
                    raise ValueError("Cannot satisfy minimum samples per client")
                donor = excess[0]
                lengths[donor] -= (min_samples_per_client - lengths[i])
                lengths[i] = min_samples_per_client

        return random_split(train_dataset, lengths)


    class_indices = defaultdict(list)
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        indices = np.array(class_indices[cls])
        if len(indices) == 0:
            continue

        np.random.shuffle(indices)


        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = np.array([p * len(indices) for p in proportions], dtype=int)


        while proportions.sum() < len(indices):
            proportions[np.random.randint(0, num_clients)] += 1
        while proportions.sum() > len(indices):
            idx = np.argmax(proportions)
            if proportions[idx] > 0:
                proportions[idx] -= 1


        start = 0
        for client_id, count in enumerate(proportions):
            end = start + count
            client_indices[client_id].extend(indices[start:end])
            start = end


    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples_per_client:

            excess_clients = [(i, len(indices)) for i, indices in enumerate(client_indices)
                              if len(indices) > min_samples_per_client]
            if not excess_clients:
                raise ValueError("Cannot satisfy minimum samples per client in Non-IID distribution")


            excess_clients.sort(key=lambda x: x[1], reverse=True)
            needed = min_samples_per_client - len(client_indices[client_id])

            for donor_id, _ in excess_clients:
                available = len(client_indices[donor_id]) - min_samples_per_client
                if available <= 0:
                    continue

                transfer = min(needed, available)
                transfer_indices = np.random.choice(client_indices[donor_id], transfer, replace=False)
                client_indices[client_id].extend(transfer_indices)
                client_indices[donor_id] = [i for i in client_indices[donor_id] if i not in transfer_indices]
                needed -= transfer
                if needed == 0:
                    break


    for indices in client_indices:
        if not indices:
            raise ValueError("One or more clients received no data")

    return [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]


import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Subset


import numpy as np
import torch
from torch.utils.data import Subset
from collections import defaultdict

def split_dataset_dirichlet_safe(train_dataset, num_clients, alpha=0.5, min_samples_per_client=10):
    total_samples = len(train_dataset)

    if min_samples_per_client * num_clients > total_samples:
        raise ValueError("Not enough samples for the requested clients")


    if hasattr(train_dataset, 'targets'):
        labels = train_dataset.targets
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    elif hasattr(train_dataset, 'samples'):
        labels = np.array([s[1] for s in train_dataset.samples])
    else:
        raise ValueError("Dataset does not have accessible labels")

    num_classes = np.max(labels) + 1
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)


    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    client_indices = [[] for _ in range(num_clients)]
    offset = 0
    for i in range(num_clients):
        client_indices[i].extend(all_indices[offset:offset + min_samples_per_client])
        offset += min_samples_per_client


    used_indices = set()
    for lst in client_indices:
        used_indices.update(lst)

    remaining_indices = list(set(range(total_samples)) - used_indices)
    np.random.shuffle(remaining_indices)
    remaining_labels = labels[remaining_indices]


    class_remain_indices = defaultdict(list)
    for i, idx in enumerate(remaining_indices):
        class_remain_indices[remaining_labels[i]].append(idx)

    for cls in range(num_classes):
        cls_idx = np.array(class_remain_indices[cls])
        if len(cls_idx) == 0:
            continue
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(cls_idx)).astype(int)


        diff = len(cls_idx) - proportions.sum()
        for i in range(abs(diff)):
            proportions[i % num_clients] += np.sign(diff)

        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(cls_idx[start:start + count])
            start += count


    client_subsets = [Subset(train_dataset, np.random.permutation(indices).tolist()) for indices in client_indices]
    return client_subsets


def add_noise_to_client(client_dataset, noise_type="gaussian", noise_level=0.1):
    if noise_type == "gaussian":

        noisy_dataset = copy.deepcopy(client_dataset)


        if isinstance(client_dataset, torch.utils.data.Subset):

            original_dataset = client_dataset.dataset
            indices = client_dataset.indices


            class NoisySubset(torch.utils.data.Dataset):
                def __init__(self, original_dataset, indices, noise_level):
                    self.original_dataset = original_dataset
                    self.indices = indices
                    self.noise_level = noise_level

                def __getitem__(self, idx):
                    real_idx = self.indices[idx]
                    x, y = self.original_dataset[real_idx]


                    if isinstance(x, torch.Tensor):
                        noise = torch.randn_like(x) * self.noise_level
                        x = x + noise

                        x = torch.clamp(x, 0, 1) if x.max() <= 1 else torch.clamp(x, 0, 255)

                    return x, y

                def __len__(self):
                    return len(self.indices)

            return NoisySubset(original_dataset, indices, noise_level)

    elif noise_type == "label":

        noisy_dataset = copy.deepcopy(client_dataset)

        if isinstance(client_dataset, torch.utils.data.Subset):
            original_dataset = client_dataset.dataset
            indices = client_dataset.indices

            class LabelNoiseSubset(torch.utils.data.Dataset):
                def __init__(self, original_dataset, indices, noise_level, num_classes=None):
                    self.original_dataset = original_dataset
                    self.indices = indices
                    self.noise_level = noise_level


                    if num_classes is None:
                        if hasattr(original_dataset, 'classes'):
                            self.num_classes = len(original_dataset.classes)
                        else:

                            self.num_classes = 10
                    else:
                        self.num_classes = num_classes

                def __getitem__(self, idx):
                    real_idx = self.indices[idx]
                    x, y = self.original_dataset[real_idx]


                    if random.random() < self.noise_level:

                        new_label = random.randint(0, self.num_classes - 1)
                        while new_label == y:
                            new_label = random.randint(0, self.num_classes - 1)
                        y = new_label

                    return x, y

                def __len__(self):
                    return len(self.indices)

            return LabelNoiseSubset(original_dataset, indices, noise_level)

    return noisy_dataset
