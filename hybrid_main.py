from env import render
from env import blob
import numpy as np
import cv2
from collections import namedtuple
import random
import math

from pytorch_nn import pytorch_nn
from pytorch_nn import memory

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


#env variables
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
SHOW_EVERY = 2000
STEPS_PER_EP = 200
episode_rewards = []
screen_height = 300
screen_width = 300
n_actions = 4

#color of agents
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255) }

#RL variables
BATCH_SIZE = 3
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#init cnn's
policy_net = pytorch_nn.DQN(screen_height, screen_width, n_actions).to(device)
target_net = pytorch_nn.DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#other things
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
optimizer = optim.RMSprop(policy_net.parameters())
memory = memory.ReplayMemory(10000, Transition)



def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        print('r')
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    print('new>>>>',batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


for episode in range(EPISODES):
    player = blob.Blob()
    food = blob.Blob()
    enemy = blob.Blob()
    last_screen = render.get_screen(food, enemy, player, d, SIZE=10, show=True)
    current_screen = render.get_screen(food, enemy, player, d, SIZE=10, show=True)
    state = current_screen - last_screen
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {EPS_START}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(STEPS_PER_EP):
        action = select_action(state, i)

        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
            print('hey1')
            next_state = None
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
            print('hey2')
            next_state = None
        else:
            last_screen = current_screen
            current_screen = render.get_screen(food, enemy, player, d, SIZE=10, show=True)
            next_state = current_screen - last_screen
            reward = -MOVE_PENALTY

        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)

        state = next_state
        optimize_model()
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break