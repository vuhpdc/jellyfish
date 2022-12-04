import numpy as np
import random
import src.simulation.env as env


### Some helper functions ###


def generate_clients(sim_data):
    clients_dict = {}
    for i in range(sim_data.num_clients):
        client = env.Client(sim_data, id=i)
        clients_dict[client.id] = client
    return clients_dict


def generate_models(sim_data, distinct=False):
    models_dict = {}
    for i in range(sim_data.num_gpus):
        idx = random.randint(0, sim_data.num_models - 1)
        while distinct and idx in models_dict:
            idx = random.randint(0, sim_data.num_models - 1)

        # idx = i + 2
        model = env.DNNModel(sim_data, model_number=idx, gpu_number=i)
        models_dict[model.id] = model
    return models_dict


def reset_clients_and_models(clients_dict, models_dict):
    for _, client in clients_dict.items():
        client.reset()
        client.next()

    for _, model in models_dict.items():
        model.reset()


def print_all_models(sim_data):
    models_dict = {}
    for i in range(sim_data.num_models):
        idx = 0
        model = env.DNNModel(sim_data, model_number=i, gpu_number=idx)
        models_dict[model.id] = model
    env.print_info({}, models_dict)


def are_floats_equal(x, y, epsilon=1e-5):
    return abs(x-y) < epsilon


def print_accuracy_ratio(algo1_accuracies, algo2_accuracies):
    accuracy_ratio = algo2_accuracies / algo1_accuracies
    print(f"\t Accuracy Ratio: Min={round(np.min(accuracy_ratio), 4)},"
          f" Mean={round(np.mean(accuracy_ratio), 4)},"
          f" Max={round(np.max(accuracy_ratio), 4)}")

    return


def print_timings(algo_timings, algo_name=""):
    print(f"\t {algo_name} Timings: Min={round(np.min(algo_timings), 4)},"
          f" Mean={round(np.mean(algo_timings), 4)},"
          f" Median={round(np.median(algo_timings), 4)},"
          f" Max={round(np.max(algo_timings), 4)}")

    return


def compare_accuracy(algo1_metric, algo2_metric, effectiveness_threshold):
    algo1_accuracy = round(algo1_metric.accuracy_per_request, 4)
    algo2_accuracy = round(algo2_metric.accuracy_per_request, 4)

    equal_count = 0
    better_count = 0
    worst_count = 0

    # Note that, the metric, named as `accuracy_per_request`,
    # takes into account the requests that are rejected.
    # Therefore, there's no need to compare the effectiveness.
    if algo2_accuracy == algo1_accuracy:
        equal_count += 1
    elif algo2_accuracy < algo1_accuracy:
        worst_count += 1
    else:
        better_count += 1

    if better_count == 1:
        print("Optimal: Acc(algo_1,algo_2)={},{}".format(algo1_accuracy,
                                                         algo2_accuracy))
    elif worst_count == 1:
        print("Worst: Acc(algo_1,algo_2)={},{}".format(algo1_accuracy,
                                                       algo2_accuracy))

    return algo1_accuracy, algo2_accuracy, equal_count, better_count, worst_count
