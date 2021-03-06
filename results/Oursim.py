import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from tqdm import trange
from environments.Oursim_Environment import make_oursim_env
from agents.DQN_agents.DDQN import DDQN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
#from agents.hierarchical_agents.HRL.HRL import HRL
#from agents.hierarchical_agents.HRL.Model_HRL import Model_HRL
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

from pandemic_simulator.environment import Hospital, PandemicSimOpts, PandemicSimNonCLIOpts, austin_regulations
from pandemic_simulator.script_helpers import test_population_params, small_town_population_params, make_gym_env, make_evaluation_plots
from pandemic_simulator.viz import MatplotLibViz
from pandemic_simulator.data import H5DataSaver, StageSchedule


config = Config()
config.seed = 100
config.num_episodes_to_run = 100
config.environment = make_oursim_env("OurSim-v0")
config.env_parameters = {}
config.file_to_save_data_results = "data_and_graphs/covid_experiments/Oursim3.pkl"
config.file_to_save_results_graph = "data_and_graphs/covid_experiments/Oursim3.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 2
config.use_GPU = False
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True


config.hyperparameters = {
    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [1, 128],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": False,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "min_steps_before_learning": 0, #TODO ? 
        "batch_size": 256,
        "discount_rate": 0.99,
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": False,
        "entropy_term_weight": 0.01,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [1, 128],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0004,
            "linear_hidden_units": [1, 128],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },
    }
}

if __name__ == "__main__":
    #AGENTS = [HRL, DDQN] #] #DDQN, ,  ] #] ##  ] #, SAC_Discrete,  SAC_Discrete, DDQN] #HRL] #, SNN_HRL, DQN, h_DQN]
    # TRAINING
    AGENTS = [SAC_Discrete]
    
    # Experimental variables
    train = False
    evaluate = False
    visualize = True
    evaluation_population = small_town_population_params
    eval_save_path = "./data_and_graphs/covid_experiments/"
    eval_save_name = "SACD_run5.pkl"
    eval_name = 'SACD'
    trainer = Trainer(config, AGENTS)

    if train:
        config.environment = make_oursim_env("OurSim-v0", population_param=evaluation_population)
        trainer.run_games_for_agents()


    # EVALUATION
    if evaluate:

        data_saver_path=Path(eval_save_path)
        data_saver = H5DataSaver(eval_name, path=data_saver_path, overwrite=True)
        config.environment = make_oursim_env("OurSim-v0", population_param=evaluation_population, evaluation=True, data_saver=data_saver)
        trainer_eval = Trainer(config, AGENTS)
        trainer_eval.evaluate_agent(SAC_Discrete)


    if visualize:
        data_saver_path=Path(eval_save_path)
        make_evaluation_plots(exp_name=eval_name, data_saver_path=data_saver_path, param_labels=['SACD'],
                          bar_plot_xlabel='Learned Strategies',
                          annotate_stages=False,
                          show_cumulative_reward=False,
                          show_time_to_peak=False, show_pandemic_duration=True,
                          )
        plt.show()
