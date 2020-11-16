from .Open_AI_Wrappers import *

import gym
import numpy as np
import pickle

from pandemic_simulator.environment import Hospital, PandemicSimOpts, PandemicSimNonCLIOpts, austin_regulations
from pandemic_simulator.script_helpers import small_town_population_params, test_population_params, make_gym_env

class OurSimCompatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_sample = self.observation(self.env.reset())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(obs_sample.size, ),
                                                dtype=np.float32)

    def observation(self, observation):
        vars = [observation.global_infection_summary.flatten(),
                observation.global_testing_summary.flatten(),
                observation.stage.flatten(),
                observation.infection_above_threshold.flatten(),
                observation.time_day.flatten(),
                ]
        return np.concatenate(vars).astype(np.float32)

class OurSimCompatActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    def action(self, action):
        return int(action)

class BasicWrapper(gym.Wrapper):
    def __init__(self, env, evaluation, eval_file_path):
        super().__init__(env)
        self.env = env
        self.evaluation=evaluation
        self.eval_file_path=eval_file_path
        self.eval_obs = []

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.evaluation:
           self.eval_obs.append((next_state, reward, done, info))

        return next_state, reward, done, info

    def reset(self):
        if self.evaluation:
           with open(self.eval_file_path, 'ab') as f:
               if len(self.eval_obs) > 0:
                       pickle.dump(self.eval_obs, f)
                       f.flush()
                       print("Saving evaluation data to file.")
        return self.env.reset()

def make_oursim_env(id="OutSimDefaultEnv-v0", population_param=test_population_params, seed=100, evaluation=False, out_path=""):
    if evaluation and out_path=="":
       print("out_path must be set when evaluating!")
       return
    numpy_rng = np.random.RandomState(seed=seed)
    max_episode_steps = 100
    # setup simulator options sets
    sim_opts = PandemicSimOpts()
    sim_non_cli_opts = PandemicSimNonCLIOpts(population_param)

    # make env
    covid_regulations = austin_regulations
    env = make_gym_env(sim_opts, sim_non_cli_opts, pandemic_regulations=covid_regulations, numpy_rng=numpy_rng)
    env = BasicWrapper(env, evaluation, out_path)
    env = OurSimCompatActionWrapper(OurSimCompatObsWrapper(env))
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env




