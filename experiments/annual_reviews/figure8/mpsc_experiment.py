'''This script runs the MPSC experiment in our Annual Reviews article.

See Figure 8 in https://arxiv.org/pdf/2108.06266.pdf.
'''

from functools import partial

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory


def run(plot=True, training=True, n_steps=30, curr_path='.'):
    '''MPSC experiment recreating Figure 8 in https://arxiv.org/pdf/2108.06266.pdf.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
    '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup PPO controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path+'/temp')

    # Load state_dict from trained PPO.
    ppo_model_dir = os.path.dirname(os.path.abspath(__file__))+'/unsafe_ppo_model'
    ctrl.load(os.path.join(ppo_model_dir,'unsafe_ppo_model_30000.pt'))  # Show violation.

    # Remove temporary files and directories
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/temp', ignore_errors=True)

    # Run without safety filter
    experiment = Experiment(env, ctrl)
    results, _ = experiment.run_evaluation(n_steps=n_steps)

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()

    if training is True:
        train_env = env_func(init_state=None, randomized_init=True)
        safety_filter.learn(env=train_env)
    else:
        safety_filter.load(f'{curr_path}/mpsc_data.pkl')

    # Run with safety filter
    experiment = Experiment(env, ctrl, safety_filter=safety_filter)
    certified_results, _ = experiment.run_evaluation(n_steps=n_steps)
    ctrl.close()
    mpsc_results = certified_results['safety_filter_data'][0]
    safety_filter.close()

    corrections = mpsc_results['correction'][0] > 1e-6
    corrections = np.append(corrections, False)

    # Plot Results
    if plot is True:
        _, ax_obs = plt.subplots()
        ax_obs.plot(certified_results['obs'][0][:, 0], certified_results['obs'][0][:, 2], '.-', label='Certified')
        ax_obs.plot(results['obs'][0][:10, 0], results['obs'][0][:10, 2], 'r--', label='Uncertified')
        ax_obs.plot(certified_results['obs'][0][corrections, 0], certified_results['obs'][0][corrections, 2], 'r.', label='Modified')
        ax_obs.legend()
        ax_obs.set_title('State Space')
        ax_obs.set_xlabel(r'$x$')
        ax_obs.set_ylabel(r'$\theta$')
        ax_obs.set_box_aspect(0.5)

        _, ax_act = plt.subplots()
        ax_act.plot(certified_results['action'][0][:], 'b-', label='Certified Inputs')
        ax_act.plot(mpsc_results['uncertified_action'][0][:], 'r--', label='Uncertified Input')
        ax_act.legend()
        ax_act.set_title('Input comparison')
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        ax_act.set_box_aspect(0.5)

        _, ax = plt.subplots()
        ax.plot(certified_results['obs'][0][:,2], certified_results['obs'][0][:,3],'.-', label='Certified')
        ax.plot(certified_results['obs'][0][corrections, 2], certified_results['obs'][0][corrections, 3], 'r.', label='Modified')
        uncert_end = results['obs'][0].shape[0]
        ax.plot(results['obs'][0][:uncert_end, 2], results['obs'][0][:uncert_end, 3], 'r--', label='Uncertified')
        ax.axvline(x=-0.2, color='r', label='Limit')
        ax.axvline(x=0.2, color='r')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    run()
