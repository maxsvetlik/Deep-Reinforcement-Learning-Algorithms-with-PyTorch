from .Open_AI_Wrappers import *

import gym
import numpy as np
import pickle

from pandemic_simulator.environment import Hospital, PandemicSimOpts, PandemicSimNonCLIOpts, austin_regulations, NoPandemicDone
from pandemic_simulator.script_helpers import small_town_population_params, test_population_params, make_gym_env
from pandemic_simulator.data import H5DataSaver, StageSchedule
from pandemic_simulator.utils import shallow_asdict

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
    def __init__(self, env, evaluation, seed, sim_opts, sim_non_cli_opts, data_saver, exp_id=0):
        super().__init__(env)
        self.env=env
        self.evaluation=evaluation
        self.data_saver=data_saver
        self.eval_obs=[]
        self.sim_opts=sim_opts
        self.sim_non_cli_opts=sim_non_cli_opts
        self.exp_id=exp_id
        self.running=False
        stages_to_execute = 0
        self.stages = ([StageSchedule(stage=stages_to_execute, end_day=None)]
            if isinstance(stages_to_execute, int) else stages_to_execute)
        self.stage_dict = {f'stage_{i}': (s.stage, s.end_day if s.end_day is not None else -1)
            for i, s in enumerate(self.stages)}
        if evaluation:
            self.env.reset()
            data_saver.begin(self.env.observation)

    def step(self, action):
        self.running=True
        next_state, reward, done, info = self.env.step(action)
        if self.evaluation:
           self.data_saver.record(next_state, reward)
           self.eval_obs.append((next_state, reward, done, info))
        return next_state, reward, done, info

    def reset(self):
        if self.evaluation and self.running:
             self.data_saver.finalize(exp_id=self.exp_id,
                               seed=100, #TODO
                               num_stages_to_execute=0, #TODO
                               num_persons=self.sim_non_cli_opts.population_params.num_persons,
                               **self.stage_dict,
                               **shallow_asdict(self.sim_opts))


        #if self.evaluation:
        #   with open(self.eval_file_path, 'ab') as f:
        #       if len(self.eval_obs) > 0:
        #               pickle.dump(self.eval_obs, f)
        #               f.flush()
        #               print("Saving evaluation data to file.")
        self.running=False
        self.exp_id += 1
        return self.env.reset()

def make_oursim_env(id="OutSimDefaultEnv-v0", population_param=test_population_params, seed=100, evaluation=False, data_saver=None):
    if evaluation and data_saver == None:
       print("data_saver_path must be set when evaluating!")
       return 0
    numpy_rng = np.random.RandomState(seed=seed)
    max_episode_steps = 100
    # setup simulator options sets
    sim_opts = PandemicSimOpts()
    sim_non_cli_opts = PandemicSimNonCLIOpts(population_param)
    # make env
    covid_regulations = austin_regulations
    env = make_gym_env(sim_opts, sim_non_cli_opts, pandemic_regulations=covid_regulations, done_fn=NoPandemicDone(30), numpy_rng=numpy_rng)
    env.reward_threshold = 0
    env.trials = 2
    env.max_episode_steps = 100
    #TODO change 100 to the actual seed used for the exp
    env = BasicWrapper(env, evaluation, 100,sim_opts, sim_non_cli_opts, data_saver)
    env = OurSimCompatActionWrapper(OurSimCompatObsWrapper(env))
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps+1)
    return env




