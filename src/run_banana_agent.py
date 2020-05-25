from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt

n_episodes = 1

env = UnityEnvironment(file_name="Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# create an Agent
agent = Agent(state_size=37, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

scores = []
scores_window = deque(maxlen=100)
for i_episode in range(0, n_episodes):
    brain_info = env.reset(train_mode=False)[brain_name]
    state = brain_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            
env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()