
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN


class TrafficEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # 0 = North-South green
        # 1 = East-West green
        self.action_space = spaces.Discrete(2)

        # State: NS queue, EW queue, current signal
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        self.max_queue = 50
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ns_queue = int(self.np_random.integers(0, 15))
        self.ew_queue = int(self.np_random.integers(0, 15))
        self.signal = 0
        self.steps = 0

        return self.get_state(), {}

    def get_state(self):
        return np.array([
            self.ns_queue / self.max_queue,
            self.ew_queue / self.max_queue,
            float(self.signal)
        ], dtype=np.float32)

    def step(self, action):
        self.signal = int(action)

        # Vehicles arrive
        self.ns_queue += int(self.np_random.poisson(2))
        self.ew_queue += int(self.np_random.poisson(2))

        # Green signal allows vehicles to leave
        if self.signal == 0:
            self.ns_queue = max(0, self.ns_queue - 5)
        else:
            self.ew_queue = max(0, self.ew_queue - 5)

        self.ns_queue = min(self.ns_queue, self.max_queue)
        self.ew_queue = min(self.ew_queue, self.max_queue)

        # Reward: fewer waiting vehicles is better
        reward = -(self.ns_queue + self.ew_queue)

        self.steps += 1

        terminated = False
        truncated = self.steps >= 100

        info = {
            "NS queue": self.ns_queue,
            "EW queue": self.ew_queue
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )


if __name__ == "__main__":
    env = TrafficEnv()

    print("Training AI traffic controller...")

    model = DQN(
        "MlpPolicy",
        env,
        learning_starts=200,
        buffer_size=5000,
        verbose=1,
        seed=42
    )

    model.learn(total_timesteps=5000)
    model.save("traffic_dqn")

    print("\nAI training complete!")

    obs, _ = env.reset(seed=10)

    print("\nTesting AI decisions:")

    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        signal = (
            "North-South GREEN"
            if action == 0
            else "East-West GREEN"
        )

        print(
            f"Step {step + 1}: {signal} | "
            f"NS queue: {info['NS queue']} | "
            f"EW queue: {info['EW queue']}"
        )

    env.close()
