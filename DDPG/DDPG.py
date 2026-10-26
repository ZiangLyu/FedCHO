import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        noise = torch.randn_like(out) * 0.1
        out = F.normalize(abs(self.fc3(out + noise)), p=1, dim=1)

        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1=256, hidden2=256, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = state
        a = action
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, args,  discount=0.99, tau=0.001):
        self.device = args.device

        state_dim = args.state_dim if hasattr(args, 'state_dim') else 140
        action_dim = args.action_dim if hasattr(args, 'action_dim') else args.aggre_num


        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay)


        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)


        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        self.actor.eval()

        expected_dim = self.actor.fc1.in_features
        if len(state) > expected_dim:
            state = state[:expected_dim]
        elif len(state) < expected_dim:
            state = np.pad(state, (0, expected_dim - len(state)), 'constant')

        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)

        return action

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()


        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)


        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()


        current_Q = self.critic(state, action)


        critic_loss = F.mse_loss(current_Q, target_Q)


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        actor_loss = -self.critic(state, self.actor(state)).mean()


        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")


    def load(self, file_name):
        self.critic.load_state_dict(torch.load(file_name + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(file_name + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):

        max_state_dim = self.state.shape[1]


        if len(state) > max_state_dim:
            state = state[:max_state_dim]
        elif len(state) < max_state_dim:
            state = np.pad(state, (0, max_state_dim - len(state)), 'constant')


        if len(next_state) > max_state_dim:
            next_state = next_state[:max_state_dim]
        elif len(next_state) < max_state_dim:
            next_state = np.pad(next_state, (0, max_state_dim - len(next_state)), 'constant')

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )