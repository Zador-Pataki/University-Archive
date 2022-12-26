from time import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from controller import Controller
from simulation import Simulation
from world import World


def write_experiment_file(experiment_results_dir, experiment_number, **variables):
    """
    SAVES TEXT FILE CONTAINING EXPERIMENT PARAMETERS AND RESULTS
    :param experiment_number (int)
    :param experiment_results_dir (string)
    :param variables: all variables that we want to store
    """
    text = ''
    for var in variables:
        text = text + str(var) + ': ' + str(variables[str(var)]) + '\n'
    text_file = open(experiment_results_dir + '/' + str(experiment_number) + "/experiment.txt", "w")
    text_file.write(text)
    text_file.close()


def save_checkpoints_plot(experiment_results_dir, experiment_number, times, n_checkpoints):
    """
    SAVES CHECKPOINTS VS TIME PLOT
    :param experiment_results_dir (string)
    :param experiment_number (int)
    :param times (list): containing time elapsed at each increment
    :param n_checkpoints (list): containing number of checkpoints at each increment
    """
    plt.plot(times, n_checkpoints)
    plt.xlabel("Number of checkpoints left")
    plt.ylabel("Simulation time")
    plt.savefig(experiment_results_dir + '/' + str(experiment_number) + '/checkpoints_v_time.png')
    plt.close()


def save_experiment_plot(n_agents, n_checkpoints_list, assignment_group_scores, experiment_results_dir):
    for assignment_group in assignment_group_scores:
        plt.plot(n_checkpoints_list, assignment_group_scores[assignment_group], label=assignment_group)
    plt.xlabel("Number of checkpoints")
    plt.ylabel("Mean simulation time")
    plt.legend()
    plt.savefig(experiment_results_dir + '/' + str(experiment_number) + '/' + str(n_agents) + '.png')
    plt.close()


if __name__ == "__main__":

    # EXPERIMENT INFO
    experiment_results_dir = 'experiments'
    experiment_number = 0
    seed = 1
    np.random.seed(seed)

    # SYSTEM PARAMETERS
    n_agents_list = [10, 10, 40, 70, 100]
    n_checkpoints_list = [250, 500, 750, 1000, 1250, 1500]
    world_size = [1000, 1000, 1000]
    view_radius = 600
    # CONTROLLER PARAMETERS
    assignment_group_list = ['clustering', 'points', 'clustering']
    cse_group_list = ['kmeans', 'spectral', 'kmeans']
    assignment_strategy_list = ['CSE', 'random', 'CSE', 'euclidean']
    cse_strategy = 3
    # SIMULATION PARAMETERS
    speed = 10
    dt = 1
    finish_process = False
    # NUMBER OF TESTS PER HYPERPARAMETER SET
    n_tests = 1

    results_matrix = np.zeros(
        (len(assignment_group_list) * len(assignment_strategy_list), len(n_agents_list) * len(n_checkpoints_list), 2))
    j = 0
    for n_agents in n_agents_list:
        for n_checkpoints in n_checkpoints_list:
            i = 0
            for assignment_group in assignment_group_list:
                for assignment_strategy in assignment_strategy_list:
                    if assignment_strategy == 'CSE' and assignment_group == 'clustering':
                        for cse_group in cse_group_list:
                            sim_times = []
                            process_times = []
                            for _ in trange(n_tests):
                                world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size,
                                              view_radius=view_radius)
                                controller = Controller(assignment_group=assignment_group,
                                                        assignment_strategy=assignment_strategy,
                                                        cse_strategy=cse_strategy, cse_group=cse_group)
                                simulation = Simulation(speed=speed, dt=dt, finish_process=finish_process)

                                start = time()
                                simulation.simulate(world, controller)
                                sim_times.append(simulation.time_elapsed_list[-1])
                                process_times.append(time() - start)

                            results_matrix[i, j, 0] = np.mean(sim_times)
                            results_matrix[i, j, 1] = np.mean(process_times)
                            i += 1
                    else:
                        sim_times = []
                        process_times = []
                        for _ in trange(n_tests):
                            world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size,
                                          view_radius=view_radius)
                            controller = Controller(assignment_group=assignment_group,
                                                    assignment_strategy=assignment_strategy, cse_strategy=cse_strategy)
                            simulation = Simulation(speed=speed, dt=dt, finish_process=finish_process)

                            start = time()
                            simulation.simulate(world, controller)
                            sim_times.append(simulation.time_elapsed_list[-1])
                            process_times.append(time() - start)

                        results_matrix[i, j, 0] = np.mean(sim_times)
                        results_matrix[i, j, 1] = np.mean(process_times)
                        i += 1
            j += 1

    if assignment_strategy == 'CSE' and assignment_group == 'clustering':
        np.save(experiment_results_dir + '/' + str(n_agents_list[0]) + '_' + assignment_group_list[0] + '_' +
                cse_group_list[
                    0] + '_' + assignment_strategy_list[0] + '_extra', results_matrix)
    else:
        np.save(experiment_results_dir + '/' + str(n_agents_list[0]) + '_' + assignment_group_list[0] + '_' +
                assignment_strategy_list[0] + '_' +
                str(seed), results_matrix)
