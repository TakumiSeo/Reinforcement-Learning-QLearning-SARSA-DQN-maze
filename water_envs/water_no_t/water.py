import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from abc import abstractmethod
import matplotlib.animation as animation
from skimage.draw import ellipse

class Water(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 3}
    FIELD = [
        'M',  # 0 agent
        'S',  # 1 start
        'G',  # 2 goal
        'W',  # 3 water
        'N',  # 4 nothing
    ]
    # this is the restriction of over iteration
    MAX_STEPS = 5000

    def __init__(self):
        super().__init__()
        self.viewer = None
        self.radius = 5
        self.rotation = 10
        self.ellipce_r = 10
        self.ellipce_c = 12
        self.x_shape = 10 * self.radius
        self.y_shape = 10 * self.radius
        self.MAP_shape = (self.x_shape, self.y_shape)
        # set an action space
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD),
            shape=self.MAP_shape
        )
        nrows, ncols = self.MAP_shape
        reward_range = [-1., 1.]
        self.reset()

    def reset(self):
        self.map = self.ellipse_map
        nrows, ncols = self.MAP_shape
        self.pos = self.find_pos('S')[0]
        # self.goal = self.find_pos('G')[0]
        self.done = False
        self.reward = 0
        self.steps = 0
        self.visited = []
        return self.observe()
        
        # dipict the map
    def ellipse_map(self):
        self.x = np.ones((self.x_shape, self.y_shape), dtype=np.uint8)
        self.x[self.x == 1] = 4
        # Start
        self.x[(0, 0)] = 1
        self.x_a, self.y_a = ellipse(self.x_shape/2, self.y_shape/2, self.ellipce_r, self.ellipce_c, rotation=np.deg2rad(self.rotation))
        self.x[(self.x_a, self.y_a)] = 3
        return self.x

    def is_movable(self, pos):
         return ((0 <= pos[0] < self.x_shape) and (0 <= pos[1] < self.y_shape))

        # judge whether agent gets to the goal
    def is_goal(self, show=False):
        nrows, ncols = self.MAP_shape
        if self.pos[0] == nrows - 1 and self.pos[1] == ncols - 1:
            if show:
                print("Goal")
            return True
        else:
            return False

    def is_done(self, show=False):
        return (not self.is_movable) or self.is_goal(show) or self.steps > self.MAX_STEPS

    def observe(self):
        # to copy the map with the place of the agent
        observation = np.copy(self.map())
        observation[tuple(self.pos)] = self.FIELD.index('M')
        return observation

    def point_finder(self):
        flat_space = np.reshape(self.observe(), [-1, 1])
        #print(flat_space)
        point = np.where(flat_space == 0)
        return int(point[0])

    def trace(self):
        self.row, self.col = np.where(self.observe() == 0)
        self.visited.append((int(self.row), int(self.col)))
        return self.visited

    def get_reward(self, pos, moved):
        nrows, ncols = self.MAP_shape
        if moved:
            if self.map()[tuple(pos)] == self.FIELD.index('W'):
                self.reward -= 10
            elif self.map()[tuple(pos)] == self.FIELD.index('N'):
                self.reward -= 0.3
        else:
            self.reward -= 0.5
        # Goal
        if self.is_goal():
            self.reward += 15
        return self.reward

    def find_pos(self, field_type):
        return np.array([np.where(self.map() == self.FIELD.index(field_type))])

    def step(self, action):
        nrows, ncols = self.MAP_shape
        if action == 0:
            next_pos = [x + y for (x, y) in zip(self.pos, [0, 1])]
        elif action == 1:
            next_pos = [x + y for (x, y) in zip(self.pos, [-1, 0])]
        elif action == 2:
            next_pos = [x + y for (x, y) in zip(self.pos, [1, 0])]
        elif action == 3:
            next_pos = [x + y for (x, y) in zip(self.pos, [-1, 0])]

        if self.is_movable(next_pos):
            self.pos = next_pos
            moved = True
        else:
            moved = False
        reward = self.get_reward(self.pos, moved)
        observation = self.observe()
        trace = self.trace()
        state = self.point_finder()
        done = self.is_done(True)
        return trace, state, reward, observation, done

    def show(self):
        # plt.grid('on')
        ims = []
        nrows, ncols = self.MAP_shape
        ax = plt.gca()
        fig = plt.figure()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.map())
        for row, col in self.visited:
            canvas[(row, col)] = self.FIELD.index('M')
            img1 = plt.imshow(canvas, interpolation="bilinear", cmap=cm.GnBu)
            ims.append([img1])
        img = plt.imshow(canvas, interpolation="bilinear", cmap=cm.GnBu, animated=True)
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
        plt.show()
        return

    @abstractmethod
    def get_image(self):
        pass

    def render(self, mode='human', max_width=500):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width / img_width
        #img = Image.fromarray(img).resize([int(ratio * img_width), int(ratio * img_height)])
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
