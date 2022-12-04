import copy
import numpy as np
import multiprocessing as mp
import sys
import os

from src.simulation.selection_algo import ModelSelectionCplexMilp, ModelSelectionSimulatedAnnealing
from src.simulation.mapping_algo import DP_on_AggregateRate as ClientMappingDP
import src.simulation.env as env
from src.simulation.opts import parse_opts
import src.simulation.utils as sim_utils


def _process_init():
    sys.stdout = open(os.devnull, 'w')


def _run_algo(algo, algo_kwargs):
    ret = algo(**algo_kwargs)
    return ret


def _run_algos_async(algo_lst, algo_args, sim_data, clients_dict, models_dict):
    '''
    Run algorithms parallelly
    '''
    results = []
    for i, algo_func in enumerate(algo_lst):
        _sim_data = copy.deepcopy(sim_data)
        _clients_dict = copy.deepcopy(clients_dict)
        _models_dict = copy.deepcopy(models_dict)

        args = {
            "simData": _sim_data,
            "initialModelsInfo": _models_dict,
            "clientsInfo": _clients_dict,
            "MappingAlgo": ClientMappingDP,
            **algo_args[i]
        }
        # _func = partial(_run_algo, algo_kwargs=args)
        # res = mp_pool.map_async(_func, [algo_func])
        res = _run_algo(algo_func, args)
        results.append(res)
    return results


def run_selection_algo(opts, algo_lst, algo_args, NUM_ITER=100):

    num_algos = len(algo_lst)
    assert num_algos == 2, "This code currenly supports comparision between only two algos"

    better_count = 0
    worst_count = 0
    equal_count = 0
    accuracies = [[] for _ in range(num_algos)]

    mp_pool = mp.Pool(num_algos, initializer=_process_init)
    iter = 0
    while iter < NUM_ITER:
        # Generate data
        sim_data = env.SimData(opts)
        clients_dict = sim_utils.generate_clients(sim_data)
        models_dict = sim_utils.generate_models(sim_data)
        mapping_info = ClientMappingDP(clients_dict, models_dict)

        if mapping_info.metrics.effectiveness < 1.0:
            # Only consider the data if there is a possiblity of mapping all clients.
            # This is for MILP. Ideally, we should know if there exists any solution for the
            # current DNNs and clients set. This should be done exhaustively.
            continue

        # Run algorithms
        sim_utils.reset_clients_and_models(clients_dict, models_dict)
        results = _run_algos_async(algo_lst,
                                   algo_args,
                                   sim_data,
                                   clients_dict,
                                   models_dict)

        algos_mapping_info = []
        for i, algo_func in enumerate(algo_lst):
            print("\n\nUsing mapping algo:", algo_func.__name__)
            # results[i].wait()
            # mapping_info, _ = results[i].get()[0]
            mapping_info, _ = results[i]
            mapping_info.print()
            mapping_info.metrics.print()

            # Deep copy is needed here, 'models_info' is shared with all algos.
            # and therefore, models in the 'models_info' will contain the updates from the
            # last algorithm.
            algos_mapping_info.append(copy.deepcopy(mapping_info))

        # Compare accuracies of two algos
        acc_algo_1, acc_algo_2, equal, better, worst = sim_utils.compare_accuracy(algos_mapping_info[0].metrics,
                                                                                  algos_mapping_info[1].metrics,
                                                                                  sim_data.effectiveness_threshold)
        if better == 0:
            # Consider the case only when the optimal algorithm has a solution to avoid division by zero.
            # algo_1 is assumed to be optimal.
            accuracies[0].append(acc_algo_1)
            accuracies[1].append(acc_algo_2)
        equal_count += equal
        better_count += better
        worst_count += worst

        print(f"Comparision: {algo_lst[0].__name__} vs {algo_lst[1].__name__}")
        print("Acc ratio: ", round(acc_algo_2 / acc_algo_1, 4))

        iter += 1

    accuracies = np.array(accuracies)
    print(accuracies)

    # Save results
    with open(f'{opts.num_gpus}_{opts.num_clients}.npy', 'wb') as f:
        np.save(f, accuracies)

    # Print some stats and results
    no_solution = NUM_ITER - (better_count + equal_count + worst_count)
    print('{:=^100}'.format(
        algo_lst[0].__name__ + " vs " + algo_lst[1].__name__))
    print("\t{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}".format("Iterations",
                                                         "Better",
                                                         "Worst",
                                                         "Equal",
                                                         "NoSolution"))
    print("\t" + "-"*100)
    print("\t{:<20d}{:<20d}{:<20d}{:<20d}{:<20d}".format(NUM_ITER,
                                                         better_count,
                                                         worst_count,
                                                         equal_count,
                                                         no_solution))

    sim_utils.print_accuracy_ratio(accuracies[0], accuracies[1])
    return


def main():
    opts = parse_opts()

    # This SA configuration seems to be good. But it takes around 321 iterations.
    # One can compute these SA parameters through sweeping.
    sa_args = {"tempInitial": 0.0125, "tempMin": 0.0005}
    milp_args = {"debug": True}
    run_selection_algo(opts,
                       algo_lst=[ModelSelectionCplexMilp,
                                 ModelSelectionSimulatedAnnealing],
                       algo_args=[milp_args, sa_args],
                       NUM_ITER=100)
    # test_cplex_milp(NUM_ITER=10)
    return


if __name__ == "__main__":
    main()
