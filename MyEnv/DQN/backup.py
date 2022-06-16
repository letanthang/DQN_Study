from collections import deque
import time
from matplotlib.pyplot import table
import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2 as cv
from Game_Environment import BlobEnv
from DQN_model import DQNmodel, QTrainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPLAY_MEMORY_SIZE = 50_000     # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64             # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5         # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200               # For model save

MAX_MEMORY = 100_000
LR = 0.001
BATCH_SIZE = 1000

# MEMORY_FRACTION = 0.20

# env settings
DISCOUNT = 0.99
EPISODES = 20_000

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.001

#stats setting
AGGREGATE_STATS_EVERY = 50 
SHOW_PREVIEW = False


env = BlobEnv()
# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
torch.random.manual_seed(1)

class DQNAgent:
    input_dim = env.OBSERVATION_SPACE_VALUES[2]
    output_dim = env.ACTION_SPACE_SIZE
    def __init__(self):
        # main model    #get trained every step
        self.model = DQNmodel(self.input_dim, self.output_dim).to(device)

        # Targer model - this is what we predict against every step
        # self.target_model = DQNmodel(input_dim, output_dim).to(device)
        # self.target_model.load_state_dict(self.model.state_dict())

        # trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=DISCOUNT)

        # An array with last n steps for training
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights   
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

     # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        # current_state = torch.Tensor(np.array(state)).reshape(-1, env.SIZE, env.SIZE)/255.0
        current_state = torch.Tensor(np.array(state)).reshape(-1, env.SIZE, env.SIZE)/255.0
        self.model.eval()
        return self.model(current_state)[0]

    # def train_long_memory(self):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = random.sample(self.replay_memory, BATCH_SIZE) # list of tuples
    #     else:
    #         mini_sample = self.replay_memory

    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    # def train_short_memory(self, state, action, reward, next_state, done):
    #     self.trainer.train_step(state, action, reward, next_state, done)    

    def learn(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch =  random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_state, action, reward, new_current_state, done = zip(*minibatch)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0].reshape(-1, env.SIZE, env.SIZE) for transition in minibatch])
        # current_states = torch.Tensor(current_states).reshape(-1,env.SIZE,env.SIZE)/255.0
        # self.model.eval()
        # current_qs_list = self.model(current_states)[0]

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3].reshape(-1, env.SIZE, env.SIZE) for transition in minibatch])
        # new_current_states = torch.Tensor(new_current_states).reshape(-1,env.SIZE,env.SIZE)/255.0
        self.trainer.train_step(current_states, action, reward, new_current_states, done) 
        # self.target_model.eval()
        # future_qs_list = self.target_model(new_current_states)[0]

        # state = []
        # reward = []

        # for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
        #     # If not a terminal state, get new q from future states, otherwise set it to 0
        #     # almost like with Q Learning, but we use just part of equation here
        #     if not done:
        #         max_future_q = np.max(future_qs_list[index])
        #         new_q = reward + DISCOUNT * max_future_q
        #     else:
        #         new_q = reward

        #      # Update Q value for given state
        #     current_qs = current_qs_list[index]
        #     current_qs[action] = new_q

        #     # And append to our training data
        #     X.append(current_state)
        #     y.append(current_qs)
        
          
        # self.model.fit (np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # updating to det if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


agent = DQNAgent()
# cs = env.reset()
# print(cs.shape)
# print(cs.reshape(-1, env.SIZE, env.SIZE).shape)

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            q_table = agent.get_qs(current_state)
            q_table = q_table.detach().numpy()
            action = np.argmax(q_table)
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.learn(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

    #     # Save model, but only when min reward is greater or equal a set value
    #     if min_reward >= MIN_REWARD:
    #         torch.save(agent.model, f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

# a = agent.model.forward(torch.randn(1,3,10,10))

# print(agent.input_dim)
# a = agent.model.forward(torch.Tensor(cs).reshape(-1,env.SIZE, env.SIZE)/255.0)

# print(a)

# def train():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
    
#     agent = DQNAgent()
#     epsilon = 1


#     for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
#         # Restarting episode - reset episode reward and step number
#         episode_reward = 0
#         step = 1

#         # Reset environment and get initial state
#         current_state = env.reset()

#         # Reset flag and start iterating until episode ends
#         done = False
#         while not done:
#             # This part stays mostly the same, the change is to query a model for Q values
#             if np.random.random() > epsilon:
#                 # Get action from Q table
#                 action = np.argmax(agent.get_qs(current_state))
#             else:
#                 # Get random action
#                 action = np.random.randint(0, env.ACTION_SPACE_SIZE)

#             new_state, reward, done = env.step(action)

#             # Transform new continous state to new discrete state and count reward
#             episode_reward += reward

#             if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
#                 env.render()

#             # Every step we update replay memory and train main network
#             agent.update_replay_memory((current_state, action, reward, new_state, done))
            
#             # agent.train_short_memory(current_state, action, reward, new_state, done)


#             current_state = new_state
#             step += 1
        
#         # Append episode reward to a list and log stats (every given number of episodes)
#         ep_rewards.append(episode_reward)
#         # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
#         #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
#         #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

#             # Save model, but only when min reward is greater or equal a set value
#             # if min_reward >= MIN_REWARD:
#                 # torch.save(agent.model, f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

#         # Decay epsilon
#         if epsilon > MIN_EPSILON:
#             epsilon *= EPSILON_DECAY
#             epsilon = max(MIN_EPSILON, epsilon)

# if __name__ == '__main__':
#     train()