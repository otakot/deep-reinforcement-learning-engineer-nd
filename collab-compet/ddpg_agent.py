import numpy as np
from collections import namedtuple, deque

from model import Actor, Critic
from ounoise import OUNoise

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from hyperparams import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, id=0,):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.id = torch.LongTensor([id])

        # Actor Network (w/ Target Network)
        # Actor has access only to its own state
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # The critic has access to all states and actions taken by all agents
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.epsilon = 1

        self.i_step = 0

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, predicted_next_actions, calculated_next_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Collect the informations from experiences
        states, actions, rewards, next_states, dones = experiences
        next_actions = torch.cat(calculated_next_actions, dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards.index_select(1, self.id) + (
            GAMMA * Q_targets_next * (1 - dones.index_select(1, self.id))
        )
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        # Minimize the loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        ###### Guess an action
        self.actor_optimizer.zero_grad()

        actions_agent = [pa if i == self.id else pa.detach() for i, pa in enumerate(predicted_next_actions)]
        actions_agent = torch.cat(actions_agent, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_agent).mean()

        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # Reduce of noise
        self.epsilon *= EPSILON_DECAY

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_weights(self):
        torch.save(self.actor_local.state_dict(), "checkpoint_actor_" + str(self.id.item()) + ".pth")
        torch.save(self.critic_local.state_dict(), "checkpoint_critic_" + str(self.id.item()) + ".pth")

    def load_weights(self):
        self.critic_local.load_state_dict(torch.load("checkpoint_critic_" + str(self.id.item()) + ".pth"))
        self.actor_local.load_state_dict(torch.load("checkpoint_actor_" + str(self.id.item()) + ".pth"))
        self.critic_target.load_state_dict(torch.load("checkpoint_critic_" + str(self.id.item()) + ".pth"))
        self.actor_target.load_state_dict(torch.load("checkpoint_actor_" + str(self.id.item()) + ".pth"))

