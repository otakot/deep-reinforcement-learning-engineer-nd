import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
from replay_buffer import ReplayBuffer

from ddpg_agent import Agent
from hyperparams import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Maddpg:
    """  Deep Deterministic Policy Gradient (MADDPG) class handles the collaboration between agents.
         It initializes n agents, handles the memory  replay buffer and orchestrates the agent's learning process . """

    def __init__(self, state_size, action_size, num_agents):
        """
        Params
        ======
            state_size(int) : dimension of the state encountered by one agent
            action_size(int): dimension of the actions taken by one agent
            num_agents(int) : number of agents in the environment
        """

        # Environment parameters
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        # Initialize the Agents. Each agent must know the dimension of its own state_size and action_size, as well as the number of other agents in the envorionment (as the critic depends on it) and its index in the agents list.
        self.agents = [Agent(state_size, action_size, num_agents, RANDOM_SEED, idx) for idx in range(0, num_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
        self.i_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experiences of agents
        self.memory.add(states.reshape(1, -1), actions, rewards, next_states.reshape(1, -1), dones)

        # Learn, if enough samples are available in memory and according to update frequency
        self.i_step = (self.i_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.i_step == 0:
            for _ in range(0, LEARN_NUM):
                # sample experiance for each agent to learn
                experiences = [self.memory.sample() for _ in range(0, self.num_agents)]
                self.learn_agents(experiences)

    def learn_agents(self, experiences):
        # actor target network is used to estimate optimal next action
        # actor local network is used to compute optimal action from the current sate
        for i, learning_agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            next_actions = []
            actions = []
            for idx, agent in enumerate(self.agents):
                agent_idx = torch.tensor([idx]).to(device)
                # get state the next state for this agent
                agent_state = (
                    states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_idx).squeeze(1)
                )
                agent_next_state = (
                    next_states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_idx).squeeze(1)
                )
                # estimate best action for next state with target actor's network
                agent_next_action = agent.actor_target(agent_next_state)
                # get best action for cucurrent state with local actor's network
                agent_action = agent.actor_local(agent_state)
                actions.append(agent_action)
                next_actions.append(agent_next_action)

            learning_agent.learn(experiences[i], actions, next_actions)

    def act(self, states):
        actions = [current_agent.act(states[i]) for i, current_agent in enumerate(self.agents)]
        return np.array(actions).reshape(1, -1)

    def reset(self):
        for a in self.agents:
            a.reset()

    def save_weights(self):
        for agent in self.agents:
            agent.save_weights()

    def load_weights(self):
        for agent in self.agents:
            agent.load_weights()

