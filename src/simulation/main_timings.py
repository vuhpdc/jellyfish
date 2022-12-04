import numpy as np
import timeit

from src.simulation.selection_algo import ModelSelectionSimulatedAnnealing
from src.simulation.mapping_algo import DP_on_AggregateRate as ClientMappingDP
import src.simulation.env as env
from src.simulation.opts import parse_opts
import src.simulation.utils as sim_utils


def test_SA_algo(opts, algo_args, NUM_ITER=100):
    completion_timings = []
    iter = 0
    while iter < NUM_ITER:
        # Generate data
        sim_data = env.SimData(opts)
        clients_dict = sim_utils.generate_clients(sim_data)
        models_dict = sim_utils.generate_models(sim_data)
        mapping_info = ClientMappingDP(clients_dict, models_dict)

        if mapping_info.metrics.effectiveness < 0.75:
            continue

        sim_utils.reset_clients_and_models(clients_dict, models_dict)
        print("\n\nUsing mapping algo:",
              ModelSelectionSimulatedAnnealing.__name__)
        start_time = timeit.default_timer()
        mapping_info, _ = ModelSelectionSimulatedAnnealing(sim_data,
                                                           models_dict,
                                                           clients_dict,
                                                           MappingAlgo=ClientMappingDP,
                                                           debug=True,
                                                           **algo_args)

        end_time = timeit.default_timer()
        elapsed_time = (end_time - start_time)*1e3
        print("Time Taken to map(ms):", elapsed_time)
        completion_timings.append(elapsed_time)

        mapping_info.print()
        mapping_info.metrics.print()
        iter += 1

    completion_timings = np.array(completion_timings)
    sim_utils.print_timings(completion_timings,
                            algo_name=ModelSelectionSimulatedAnnealing.__name__)


def main():
    opts = parse_opts()

    # This SA configuration seems to be good. But it takes around 321 iterations.
    # One can compute these SA parameters through sweeping.
    sa_args = {"tempInitial": 0.0125, "tempMin": 0.0005}
    test_SA_algo(opts, sa_args, NUM_ITER=1000)

    return


if __name__ == "__main__":
    main()
