# udacity-drlnd-navigation

# Introduction
This is my submission for the Navigation project of the Udacity Deep Reinforcement Learning Nanodegree.  The purpose of the project is to train an agent to collect yellow bananas while avoiding blue bananas in a Unity environment, using an adaptation of the Deep Q-learning (DQN) algorithm.

# Project Details
The environment for this project is a Unity ML-Agents environment provided by Udacity.  In this environment, an agent navigates a square world filled with yellow bananas and blue bananas.  The agent is provided a reward of +1 for collecting a yellow banana, and a reward of -1 for collecting a blue banana.  Therefore, the agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space (of the agent's observations) is a vector of 37 numbers that contains information on the agent's velocity, as well as ray-based perception of objects in the agent's forward field-of-view.  

The agent's action space is discrete, with 4 possible actions at each step:
* 0: move forward
* 1: move backward
* 2: turn left
* 3: turn right

The environment is considered solved when the agent obtains an average score of at least +13 over 100 consecutive episodes.

# Getting Started
The dependencies for this submission are the same as for the [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies):
* Python 3.6
* pytorch
* unityagents
* [Udacity banana project environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

This project was completed and tested on a local Windows environment (64-bit) running Python 3.6.10 in a conda environment.

# Instructions
To train the agent, run **train_banana_agent.py**.  This will save the model parameters in **checkpoint.pth** once the agent has fulfilled the criteria for considering the environment solved.

To run the trained agent, run **run_banana_agent.py**.  This will load the saved model parameters from checkpoint.pth, and run the trained model in the banana-collecting environment.  The *n_episodes* parameter is the number of episodes that will be run.  By default, this parameter is set to 100 to facilitate validation of the trained agent.
