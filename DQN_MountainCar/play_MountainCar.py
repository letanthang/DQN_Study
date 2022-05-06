import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import matplotlib.animation as animation

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 3
SHOW_EVERY = 1

epsilon = 0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES 
# END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

env = gym.make("MountainCar-v0")
env.reset()

q_table = np.load(f"D:/Code/Python/Study_AI/ML/DQN_Study/DQN_MountainCar/qtables/24990-qtable.npy")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int_))


fig = plt.figure()
print(env.action_space,env.observation_space)
ims = []

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            #get rand action
            action = np.random.randint(0, env.action_space.n)
    
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            img = env.render(mode='rgb_array')
            cv2_im_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            img = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
            im = plt.imshow(img, animated=True)
            ims.append([im])
        if not done:
            # Update Q table with new Q value
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action, )]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value - after taken the action
            q_table[discrete_state + (action, )] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly	
        elif new_state[0] >= env.goal_position:
            # print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)

    # if not episode % 10:
    # 	np.save(f"qtables/{episode}-qtable.npy", q_table)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward) 
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:])) 
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:])) 

        print(f'Episode: {episode}, average: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}')

env.close()
Writer = animation.writers['pillow']
writer = Writer(fps=30 , metadata=dict(artist='Me'), bitrate=1800)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=True)
im_ani.save('DQN_mountaincar_1.gif', writer=writer)    