import gym
from gym import spaces
import numpy as np
import serial  # To read data from Arduino

class EEGWheelchairEnv(gym.Env):
    def __init__(self):
        super(EEGWheelchairEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Stop, 1: Forward, 2: Backward
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initializing connection to Arduino (assumes serial connection)
        self.serial_port = serial.Serial('COM3', 9600)  # Adjust COM port as necessary
        self.state = np.zeros(3)  # Placeholder for EEG signal
        self.done = False

    def reset(self):
        self.state = np.zeros(3)
        self.done = False
        return self.state

    def step(self, action):
        reward = self.compute_reward(action)
        self.state = self.get_next_state()
        self.done = self.check_done()
        return self.state, reward, self.done, {}

    def compute_reward(self, action):
        # Customize the reward function based on specific goals (e.g., moving forward)
        if action == 1:  # Forward
            return 1.0
        elif action == 2:  # Backward
            return -1.0
        else:  # Stop
            return -0.1

    def get_next_state(self):
        # Read data from Arduino (EEG signals) and normalize
        try:
            eeg_data = self.serial_port.readline().decode('utf-8').strip()
            eeg_signals = np.array([float(i) for i in eeg_data.split(",")])
        except:
            eeg_signals = np.zeros(3)  # In case of read failure, return zeros
        return eeg_signals

    def check_done(self):
        return np.random.rand() > 0.98  # Termination condition

env = EEGWheelchairEnv()
