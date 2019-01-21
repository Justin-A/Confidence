import time
import numpy as np
import random
import copy
HEIGHT = 10
WIDTH = 10

HUMAN_num = 3
OIL_num = 3
REEF_num = 0


class Env():
    def __init__(self):
        super().__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)

        self.blue_list = []
        self.trail = []

        self.reset_time = 0
        self.delay_time = 0
        self.counter = 0
        self.rewards = []
        self.goal = []

        self.state_size = 0

        self.first_boat = [0, 0]
        self.reef_reward = -100
        self.oil_reward = 1
        self.human_reward = 10

        self.oil_time_reward = -0.01
        self.human_time_reward = -0.1

        self.first_states = []
        self.first_oil_states = [6, 6]
        self.WH_list = [[i, j] for i in range(WIDTH) for j in range(HEIGHT)]
        for i in range(5):
            for j in range(5):
                self.WH_list.remove([i, j])

    def reset_reward(self):
        self.rewards.clear()
        oil_random = True
        self.first_states = []

        self.first_states_list = random.sample(self.WH_list, REEF_num + OIL_num + HUMAN_num)

        # Reset
        # Reef
        for i in range(REEF_num):
            self.set_reward(self.first_states_list[i], self.reef_reward)

        # OIL
        for i in range(OIL_num):
            if oil_random == True:
                self.set_reward(self.first_states_list[REEF_num], self.oil_reward) # REEF_num or i + REEF_num
            elif oil_random == False:
                self.set_reward(self.first_oil_states, self.oil_reward)
                #self.first_coords_list[i + REEF_num] = self.rewards[i + REEF_num]['coords']
        # HUMAN
        for i in range(HUMAN_num):
            self.set_reward(self.first_states_list[i + REEF_num + OIL_num], self.human_reward)

        # Write first coordinates
        for i in range(REEF_num + OIL_num + HUMAN_num):
            self.first_states.append(self.rewards[i]['state'])

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward == self.reef_reward:
            temp['reward'] = reward

        elif reward == self.oil_reward:
            temp['reward'] = reward

        elif reward == self.human_reward:
            temp['reward'] = reward

        temp['state'] = state
        self.rewards.append(temp)  # 'reward' : (-999, 10, 100), 'figure' : (key value), 'coords' : (UNIT, UNIT), 'state' : (x, y)

    def check_if_reward(self, state):  # Check terminal state, reward
        check_list = dict()
        # check_list['Episode_Finish'] = False
        check_list['obstacle'] = False
        check_list['No Object'] = False

        rewards = 0
        for reward in self.rewards:
            if reward['reward'] == self.oil_reward:
                rewards += self.oil_time_reward
            if reward['reward'] == self.human_reward:
                rewards += self.human_time_reward


        for reward in self.rewards:  # 'reward' : (-999, 10, 100), 'figure' : (key value), 'coords' : (UNIT, UNIT), 'state' : (x, y)
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] in [self.oil_reward, self.human_reward, -1]:
                    reward['reward'] = 0
                elif reward['reward'] == self.reef_reward:
                    check_list['obstacle'] = True

        cnt = 0
        for x in self.rewards:
            if x['reward'] == 0:
                cnt += 1

        if cnt == (HUMAN_num + OIL_num):
            check_list['No Object'] = True

        check_list['rewards'] = rewards

        return check_list

    def reset(self):
        if self.reset_time !=0:
            time.sleep(self.reset_time)
        self.boat=[0,0]
        self.reset_reward()
        # self.life = 1
        return self.get_state()

    def move(self, target, action):
        s = target

        if action == 0:  # Up
            if s[1] > 0:
                s[1] -= 1
        elif action == 1:  # Down
            if s[1] < (HEIGHT - 1):
                s[1] += 1
        elif action == 2:  # Left
            if s[0] < (WIDTH - 1):
                s[0] += 1
        elif action == 3:  # Right
            if s[0] > 1:
                s[0] -= 1
        s_ = s

        return s_

    def get_state(self):
        location = self.boat
        agent_x = location[0]
        agent_y = location[1]

        states = list()
        for reward in self.rewards:  # List of oil, reef, human
            tmp = []
            reward_location = reward['state']
            tmp.append(reward_location[0] - agent_x)
            tmp.append(reward_location[1] - agent_y)

            if reward['reward'] == self.reef_reward:
                tmp.append(self.reef_reward)  # reef -> -1
            elif reward['reward'] == self.oil_reward:
                tmp.append(self.oil_reward)  # oil -> 1
            elif reward['reward'] == self.human_reward:
                tmp.append(self.human_reward)  # human -> 2
            else:
                tmp.append(0)
            states.append(tmp)

        for i in range(len(states)):
            if states[i][2] == 0:
                states[i][0] = 0
                states[i][1] = 0

        self.state_size = len(states)

        return states

    def move_rewards(self):
        new_rewards = []
        states_list = []
        for states_temp in self.rewards:
            states_list.append(states_temp['state'])

        for i, temp in enumerate(self.rewards):
            if temp['reward'] == self.reef_reward:
                new_rewards.append(temp)
                continue
            elif temp['reward'] == self.human_reward:
                temp['state'] = self.move_human(temp, i, states_list)
            elif temp['reward'] == self.oil_reward:
                temp['state'] = self.move_const(temp, states_list)
            states_list[i] = temp['state']
            new_rewards.append(temp)
        return new_rewards

    def move_const(self, target, states_list):
        direction_space = copy.deepcopy(self.action_space)  # self.action_space = ['u', 'd', 'l', 'r']
        rel_coords = set(self.tuples([[j[0] - target['state'][0], j[1] - target['state'][1]] for j in
                                      states_list]))  # coords_list = [[x1,y1]-s, [x2,y2]-s, [x3,y3]-s, ...]

        for li in rel_coords:
            if li == (1, 0):
                direction_space.remove('r')
            elif li == (-1, 0):
                direction_space.remove('l')
            elif li == (0, 1):
                direction_space.remove('d')
            elif li == (0, -1):
                direction_space.remove('u')

        if (target['state'][0] < 1) and ('l' in direction_space):
            direction_space.remove('l')
        elif (target['state'][0] > (WIDTH - 1)) and ('r' in direction_space):
            direction_space.remove('r')
        if (target['state'][1] < 1) and ('u' in direction_space):
            direction_space.remove('u')
        elif (target['state'][1] > (HEIGHT - 1)) and ('d' in direction_space):
            direction_space.remove('d')

        if len(direction_space) > 0:
            direction = random.sample(direction_space, 1)
        else:
            direction = ['stop']

        if direction == ['r']:
            target['state'][0] += 1
        elif direction == ['l']:
            target['state'][0] -= 1
        elif direction == ['d']:
            target['state'][1] += 1
        elif direction == ['u']:
            target['state'][1] -= 1
        elif direction == ['stop']:
            pass

        return target['state']


    def move_human(self, target, i, states_list):
        direction_space = copy.deepcopy(self.action_space)
        direction=[]
        rel_coords = set(self.tuples([[j[0] - target['state'][0], j[1] - target['state'][1]] for j in
                                      states_list]))

        for li in rel_coords:
            if li == (1, 0):
                direction_space.remove('r')
            elif li == (-1, 0):
                direction_space.remove('l')
            elif li == (0, 1):
                direction_space.remove('d')
            elif li == (0, -1):
                direction_space.remove('u')

        if (target['state'][0] < 1) and ('l' in direction_space):
            direction_space.remove('l')
        elif (target['state'][0] > (WIDTH - 1)) and ('r' in direction_space):
            direction_space.remove('r')
        if (target['state'][1] < 1) and ('u' in direction_space):
            direction_space.remove('u')
        elif (target['state'][1] > (HEIGHT - 1)) and ('d' in direction_space):
            direction_space.remove('d')

        rel_first = [self.first_states[i][0] - target['state'][0], self.first_states[i][1] - target['state'][1]]

        if rel_first == [1, 0]:
            direction = ['r']
        elif rel_first == [-1, 0]:
            direction = ['l']
        elif rel_first == [0, 1]:
            direction = ['d']
        elif rel_first == [0, -1]:
            direction = ['u']
        elif rel_first == [0,0]:
            if len(direction_space) > 0:
                direction = random.sample(direction_space, 1)
            else:
                direction = ['stop']

        if direction == ['r']:
            target['state'][0] += 1
        elif direction == ['l']:
            target['state'][0] -= 1
        elif direction == ['d']:
            target['state'][1] += 1
        elif direction == ['u']:
            target['state'][1] -= 1
        elif direction == ['stop']:
            pass

        s_ = target['state']
        return s_

    def step(self, action):
        self.counter += 1
        self.render()

        # self.rewards = self.move_rewards()
        if self.counter % 10 == 0:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.boat, action)  # Boat Action ; return coords(x), coords(y)
        check = self.check_if_reward(next_coords)  # Action 일 때 Check Reward
        done = check['obstacle'] + check['No Object']
        # done = check['Episode_Finish'] + check['obstacle'] # True / False + True / False
        goal = check['No Object']  # True / False
        reward = check['rewards']


        remain_human = 0
        remain_oil = 0
        for reward_temp in self.rewards:
            if reward_temp['reward'] == self.human_reward:
                remain_human += 1
            if reward_temp['reward'] == self.oil_reward:
                remain_oil += 1

        s_ = self.get_state()

        return s_, reward, done, goal, remain_human, remain_oil  # state (x, y), reward (reef, oil, human), True / False, True / False

    def tuples(self, A):
        try:
            return tuple(self.tuples(a) for a in A)
        except TypeError:
            return A

    def render(self):
        if self.delay_time != 0:
            time.sleep(self.delay_time)