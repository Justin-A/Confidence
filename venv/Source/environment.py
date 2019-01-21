import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import random
import copy
PhotoImage = ImageTk.PhotoImage
UNIT = 30
HEIGHT = 10
WIDTH = 10

HUMAN_num = 3
OIL_num = 3
REEF_num = 0


class Env(tk.Tk):
    def __init__(self):
        super().__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title("Confidence")
        self.geometry("{0}x{1}".format(WIDTH * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()

        self.blue_list = []
        self.trail = []

        self.reset_time = 0
        self.delay_time = 0
        self.counter = 0
        self.rewards = []
        self.goal = []

        self.state_size = 0

        self.reef_reward = -100
        self.oil_reward = 1
        self.human_reward = 10

        self.oil_time_reward = -0.01
        self.human_time_reward = -0.1

        self.first_coords = []
        self.first_oil_coords = [6, 6]
        self.WH_list = [[i, j] for i in range(WIDTH) for j in range(HEIGHT)]
        for i in range(5):
            for j in range(5):
                self.WH_list.remove([i, j])

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='cornflower blue', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        # for c in range(0, WIDTH * UNIT, UNIT):
        #    x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
        #    canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, HEIGHT * UNIT, UNIT):
        #    x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
        #    canvas.create_line(x0, y0, x1, y1)

        # Boat
        x, y = (WIDTH - 1) * (UNIT / 2), (HEIGHT - 1) * (UNIT / 2)  # Index
        self.boat = canvas.create_image(x, y, image=self.shapes[0]) # x, y

        canvas.pack()

        return canvas

    def load_images(self):
        # boat.jpg, oil.png, reef.jpg, sos.png
        boat = PhotoImage(Image.open("../IMG/boat.png").resize((UNIT, UNIT)))
        reef = PhotoImage(Image.open("../IMG/reef.png").resize((UNIT, UNIT)))
        oil = PhotoImage(Image.open("../IMG/oil.png").resize((UNIT, UNIT)))
        human = PhotoImage(Image.open("../IMG/sos.png").resize((UNIT, UNIT)))
        blue = PhotoImage(Image.open("../IMG/blue.png").resize((UNIT, UNIT)))

        return boat, reef, oil, human, blue

    # def reset_reward(self):
    #     for reward in self.rewards:
    #         self.canvas.delete(reward['figure'])
    #
    #     self.rewards.clear()
    #
    #     # Reset
    #
    #     # Reef
    #     for i in range(REEF_num):
    #         self.set_reward([random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)], self.reef_reward)
    #
    #     # OIL
    #     for i in range(OIL_num):
    #         self.set_reward([10, 10], self.oil_reward)
    #
    #     # HUMAN
    #     for i in range(HUMAN_num):
    #         self.set_reward([random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)], self.human_reward)

    def reset_reward(self):
        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        oil_random = True
        self.first_coords = []

        self.first_coords_list = random.sample(self.WH_list, REEF_num + OIL_num + HUMAN_num)

        # Reset
        # Reef
        for i in range(REEF_num):
            self.set_reward(self.first_coords_list[i], self.reef_reward)

        # OIL
        for i in range(OIL_num):
            if oil_random == True:
                self.set_reward(self.first_coords_list[REEF_num], self.oil_reward) # REEF_num or i + REEF_num
            elif oil_random == False:
                self.set_reward(self.first_oil_coords, self.oil_reward)
                #self.first_coords_list[i + REEF_num] = self.rewards[i + REEF_num]['coords']
        # HUMAN
        for i in range(HUMAN_num):
            self.set_reward(self.first_coords_list[i + REEF_num + OIL_num], self.human_reward)

        # Write first coordinates
        for i in range(REEF_num + OIL_num + HUMAN_num):
            self.first_coords.append(self.rewards[i]['coords'])

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward == self.reef_reward:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])

        elif reward == self.oil_reward:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[2])

        elif reward == self.human_reward:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])

        temp['coords'] = self.canvas.coords(temp['figure'])
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
                    self.canvas.delete(reward['figure'])
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

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * UNIT + UNIT / 2)
        y = int(state[1] * UNIT + UNIT / 2)
        return [x, y]

    def reset(self):
        if len(self.blue_list) > 0:
            for i in range(len(self.blue_list)):
                self.canvas.delete(i+self.blue_list[0])
        self.blue_list = []
        self.trail = []

        self.update()
        if self.reset_time !=0:
            time.sleep(self.reset_time)
        x, y = self.canvas.coords(self.boat)
        self.canvas.move(self.boat, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        # self.life = 1
        return self.get_state()

    def move(self, target, action):
        s = self.canvas.coords(target)
        base_action = np.array([0, 0])

        if action == 0:  # Up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # Down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # Left
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # Right
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(target, base_action[0], base_action[1])
        s_ = self.canvas.coords(target)
        return s_

    def get_state(self):
        location = self.coords_to_state(self.canvas.coords(self.boat))
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

        return states  # {agent_location (x, y) + (-1 or 1 or 2)} * reward_num

    # def move_rewards(self):
    #     new_rewards = []
    #     for temp in self.rewards:
    #         if temp['reward'] != self.oil_reward:
    #             new_rewards.append(temp)
    #             continue
    #         temp['coords'] = self.move_const(temp)
    #         temp['state'] = self.coords_to_state(temp['coords'])
    #         new_rewards.append(temp)
    #     return new_rewards

    def move_rewards(self):
        new_rewards = []
        coords_list = []
        for coords_temp in self.rewards:
            coords_list.append(coords_temp['coords'])

        for i, temp in enumerate(self.rewards):
            if temp['reward'] == self.reef_reward:
                new_rewards.append(temp)
                continue
            elif temp['reward'] == self.human_reward:
                temp['coords'] = self.move_human(temp, i, coords_list)
            elif temp['reward'] == self.oil_reward:
                temp['coords'] = self.move_const(temp, coords_list)
            coords_list[i] = temp['coords']
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
        return new_rewards


    # def move_const(self, target):
    #     base_action = np.array([0, 0])
    #     ind = random.randint(0, 3)
    #     s = target['coords']
    #     if ind == 0:
    #         if s[0] < (WIDTH - 1) * UNIT:
    #             base_action[0] += UNIT
    #     elif ind == 1:
    #         if s[0] > UNIT:
    #             base_action[0] -= UNIT
    #     elif ind == 2:
    #         if s[1] < (HEIGHT - 1) * UNIT:
    #             base_action[1] += UNIT
    #     else:
    #         if s[1] > UNIT:
    #             base_action[1] -= UNIT
    #
    #     self.canvas.move(target['figure'], base_action[0], base_action[1])
    #     s_ = self.canvas.coords(target['figure'])
    #     return s_

    def move_const(self, target, coords_list):
        direction_space = copy.deepcopy(self.action_space)  # self.action_space = ['u', 'd', 'l', 'r']
        rel_coords = set(self.tuples([[j[0] - target['coords'][0], j[1] - target['coords'][1]] for j in
                                      coords_list]))  # coords_list = [[x1,y1]-s, [x2,y2]-s, [x3,y3]-s, ...]

        for li in rel_coords:
            if li == (UNIT, 0):
                direction_space.remove('r')
            elif li == (-UNIT, 0):
                direction_space.remove('l')
            elif li == (0, UNIT):
                direction_space.remove('d')
            elif li == (0, -UNIT):
                direction_space.remove('u')

        if (target['coords'][0] < UNIT) and ('l' in direction_space):
            direction_space.remove('l')
        elif (target['coords'][0] > (WIDTH - 1) * UNIT) and ('r' in direction_space):
            direction_space.remove('r')
        if (target['coords'][1] < UNIT) and ('u' in direction_space):
            direction_space.remove('u')
        elif (target['coords'][1] > (HEIGHT - 1) * UNIT) and ('d' in direction_space):
            direction_space.remove('d')

        if len(direction_space) > 0:
            direction = random.sample(direction_space, 1)
        else:
            direction = ['stop']

        if direction == ['r']:
            self.canvas.move(target['figure'], UNIT, 0)
        elif direction == ['l']:
            self.canvas.move(target['figure'], -UNIT, 0)
        elif direction == ['d']:
            self.canvas.move(target['figure'], 0, UNIT)
        elif direction == ['u']:
            self.canvas.move(target['figure'], 0, -UNIT)
        elif direction == ['stop']:
            self.canvas.move(target['figure'], 0, 0)

        s_ = self.canvas.coords(target['figure'])
        return s_


    def move_human(self, target, i, coords_list):
        direction_space = copy.deepcopy(self.action_space)
        direction=[]
        rel_coords = set(self.tuples([[j[0] - target['coords'][0], j[1] - target['coords'][1]] for j in
                                      coords_list]))

        for li in rel_coords:
            if li == (UNIT, 0):
                direction_space.remove('r')
            elif li == (-UNIT, 0):
                direction_space.remove('l')
            elif li == (0, UNIT):
                direction_space.remove('d')
            elif li == (0, -UNIT):
                direction_space.remove('u')

        if (target['coords'][0] < UNIT) and ('l' in direction_space):
            direction_space.remove('l')
        elif (target['coords'][0] > (WIDTH - 1) * UNIT) and ('r' in direction_space):
            direction_space.remove('r')
        if (target['coords'][1] < UNIT) and ('u' in direction_space):
            direction_space.remove('u')
        elif (target['coords'][1] > (HEIGHT - 1) * UNIT) and ('d' in direction_space):
            direction_space.remove('d')

        rel_first = [self.first_coords[i][0] - target['coords'][0], self.first_coords[i][1] - target['coords'][1]]

        if rel_first == [UNIT, 0]:
            direction = ['r']
        elif rel_first == [-UNIT, 0]:
            direction = ['l']
        elif rel_first == [0, UNIT]:
            direction = ['d']
        elif rel_first == [0, -UNIT]:
            direction = ['u']
        elif rel_first == [0,0]:
            if len(direction_space) > 0:
                direction = random.sample(direction_space, 1)
            else:
                direction = ['stop']

        if direction == ['r']:
            self.canvas.move(target['figure'], UNIT, 0)
        elif direction == ['l']:
            self.canvas.move(target['figure'], -UNIT, 0)
        elif direction == ['d']:
            self.canvas.move(target['figure'], 0, UNIT)
        elif direction == ['u']:
            self.canvas.move(target['figure'], 0, -UNIT)
        elif direction == ['stop']:
            self.canvas.move(target['figure'], 0, 0)

        s_ = self.canvas.coords(target['figure'])
        return s_

    def step(self, action):
        self.counter += 1
        self.render()

        # self.rewards = self.move_rewards()
        if self.counter % 10 == 0:
            self.rewards = self.move_rewards()

        # blue
        tmp = self.canvas.coords(self.boat)
        if tmp not in self.trail:
            tmp2 = self.canvas.create_image(tmp[0], tmp[1], image=self.shapes[4])
            self.blue_list.append(tmp2)  # 지나간 자리에 blue 이미지 누적
            self.canvas.tag_lower(tmp2)
        self.trail.append(tmp)
        next_coords = self.move(self.boat, action)  # Boat Action ; return coords(x), coords(y)
        check = self.check_if_reward(self.coords_to_state(next_coords))  # Action 일 때 Check Reward
        done = check['obstacle'] + check['No Object']
        # done = check['Episode_Finish'] + check['obstacle'] # True / False + True / False
        goal = check['No Object']  # True / False
        reward = check['rewards']

        #if self.trail[-1] in self.trail[-3:-1]:
        #    reward -= 0.1
            #print(self.trail[-3:-1], self.trail[-1])

        remain_human = 0
        remain_oil = 0
        for reward_temp in self.rewards:
            if reward_temp['reward'] == self.human_reward:
                remain_human += 1
            if reward_temp['reward'] == self.oil_reward:
                remain_oil += 1

        self.canvas.tag_raise(self.boat)  # Canvas 내 Boat 움직임
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
        self.update()