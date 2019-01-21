import matplotlib
matplotlib.use('TkAgg')

import copy
import pylab
import random
import numpy as np

from environment import Env # Show canvas
# from environment2 import Env # For Train

from keras.layers import Dense, Conv1D
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 3000

class A2C_Agent:
    def __init__(self, state_size):
        self.load_model = False
        if self.load_model:
            self.actor.load_weights('./save_model/actor.h5')
            self.critic.load_weights('./save_model/critic.h5')
        
        self.staate_size = state_size
        self.discount_factor = 0.99
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.value_size = 1

        # A2C Hyper Parameters
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

        # Build Model
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()
    
    # Build Actor Network
    def build_actor(self):
        actor = Sequential()
        actor.add(Conv1D(filters = 10, kernel_size = 1, input_shape = (self.state_size, 3), kernel_initializer = 'he_uniform', activation = 'tanh'))
        actor.add(Dense(30, kernel_initializer = 'he_uniform', activation = 'tanh'))
        actor.add(Dense(30, kernel_initializer = 'hu_uniform', activation = 'tanh'))
        actor.add(Dense(self.action_size, activation = 'softmax')) # Output : Policy
        actor.summary()
        return actor
    
    # Build Critic Network
    def build_critic(self):
        critic = Sequential()
        critic.add(Conv1D(filters = 10, kernel_size = 1, input_shape = (self.state_size, 3), kernel_initializer = 'he_uniform', activation = 'tanh'))
        critic.add(Dense(30, kernel_initializer = 'he_uniform', activation = 'tanh'))
        critic.add(Dense(30, kernel_initializer = 'he_uniform', activation = 'tanh'))
        critic.add(Dense(self.value_size, activation = 'linear'))
        return critic
    
    # Get action by Output of actor network
    def get_action(self, state):
        policy = seelf.actor.predict(state, batch_size = 1).flatten()
        return np.random.choice(self.action_size, 1, p = policy)[0], policy

    # Optimize Actor Network
    def actor_optimizer(self):
        action = K.placeholder(shape = [None, self.action_size])
        advantage = K.placeholder(shape = [None, ])

        action_prob = K.sum(action * self.actor.output, axis = 1)
        cross_entropy = K.log(action_prob + 1e-10) * advantage
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(self.actor.output * K.log(self.actor.output + 1e-10), axis = 1)
        loss = cross_entropy + 0.01
        
        optimizer = Adam(lr = self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [], updates = updates)
        return train
    
    # Optimize Critic Network
    def critic_optimizer(self):
        target = K.placeholder(shape = [None, ])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr = self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input,target], [], updates = updates)
        return train
    
    # Train model by time-step
    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # Update Advantage with target
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value
        
        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])

if __name__ == "__main__":
    env = Env()
    state = env.reset()
    state_size = env.state_size

    agent = A2C_Agent(state_size)
    
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        score, step, done = 0, 0, False
        state = env.reset()
        state = np.expand_dims(state, axis = 0)
        goal, total_dead, total_remain_oil = 0, 0, 0

        while not done:
            global_step += 1
            step += 1

            action, policy = agent.get_action(state)
            if global_step == 1:
                policy = [0.25, 0.25, 0.25, 0.25]
            
            if step > 500:
                action = random.randint(0, 3)
            
            next_state, reward, done, goal, remain_human, remain_oil = env.step(action)

            if step % 100 == 0:
                value = agent.critic.predict(state)[0]
                next_state_temp = np.expand_dims(next_state, axis = 0)
                next_value =agent.critic.predict(next_state_temp)[0]
                advantage = (reward + agent.discount_factor * next_value) - value
                print("Policy: ", policy, "Action: ", action, "Advantage: ", advantage)
            
            next_state = np.expand_dims(next_state, axis = 0)
            agent.train_model(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                if goal == 1:
                    goal = "Goal!"
                else:
                    goal = "No Goal..."
                    total_dead += remain_human
                    total_remain_oil += remain_oil

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'r')
                pylab.savefig("./save_graph/A2C_Agent.png")
                print("episode: ", e, "score: ", score, "step: ", step, "global_step: ", global_step, "Goal: ", goal, "Total_Dead: ", total_dead, "Total_Remain_Oil: ", total_remain_oil)
        
        # Save weight
        if e % 10 == 0:
            agent.actor.save_weights("./save_model/actor.h5")
            agent.critic.save_weights("./save_model/critic.h5")
            
    print('Mean Step : ', (global_step / EPISODES), 'Survival rate : ', (1 - total_dead / (3 * EPISODES)),
          'Collection rate : ', (1 - total_remain_oil / (3 * EPISODES)))
