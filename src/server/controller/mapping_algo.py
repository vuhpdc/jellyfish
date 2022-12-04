import numpy as np
from src.server.controller.utils import MappingInfo


'''
The best algorithm for mapping users to clients.
'''


def DP_on_AggregateRate(clients_info, models_info):
    # Sorting key functions
    def client_sort_key(x, model_number):
        return x.lat_budget[model_number]

    def model_sort_key(x):
        return x.accuracy

    def get_max_batch_size(client, model, max_batch_size=12):
        lat = client.lat_budget[model.model_number]
        b = max_batch_size
        # TODO: Use binary search.
        '''
        Note that, we should stop at a batch point when throughput curve starts dropping.
        As there is no benefit in continuing beyond this batch point even though subsequent 
        batch points might have higher throughput. It breaks our assumption that higher batch sizes 
        have higher throughput and therefore, our DP iteration might break.
        '''
        while b >= 1 and lat < 2 * model.latency_vec[b-1]:
            b -= 1
        batch_lat = b

        b = 0
        while b < (max_batch_size - 1):
            if model.throughput_vec[b] > model.throughput_vec[b+1]:
                break
            b += 1
        batch_throughput = b + 1

        return min(batch_throughput, batch_lat)

    def construct_dp_mat(model, sorted_clients):
        '''
        Ref: http://cse.unl.edu/~goddard/Courses/CSCE310J/Lectures/Lecture8-DynamicProgramming.pdf
        '''

        # Allocate DP matrix based on maximum batch size
        max_batch_size = get_max_batch_size(sorted_clients[0], model)
        if max_batch_size <= 0:
            return None, None, None

        assert max_batch_size > 0, "Model cannot support any sorted client"
        max_throughput = model.throughput_vec[max_batch_size-1]
        max_throughput_enum = int(max_throughput // throughput_divisor)

        # TODO: Looks like, the dp_mat can be initialized  with Zeros
        dp_mat = np.full(
            (total_clients+1, max_throughput_enum+1), -1, dtype=int)
        dp_mat[0, :] = 0
        dp_mat[:, 0] = 0

        # Construct dp matrix for each client row
        best_value = 0
        best_cell = (0, 0)
        for i, client in enumerate(sorted_clients, start=1):
            max_batch_size = get_max_batch_size(client, model, max_batch_size)
            if max_batch_size <= 0:
                break
            max_throughput = model.throughput_vec[max_batch_size-1]
            max_throughput_enum = int(max_throughput // throughput_divisor)

            # NOTE: standard DP algo. Careful with the indexing.
            w_i = client.fps  # Weight
            v_i = client.fps  # Value
            for j in range(1, max_throughput_enum+1):
                w = j * throughput_divisor
                if w_i > w:
                    dp_mat[i, j] = dp_mat[i-1, j]
                else:
                    j_diff = (w - w_i) // throughput_divisor
                    new_value = v_i + dp_mat[i-1, j_diff]
                    assert new_value >= 0 and j_diff >= 0

                    if (new_value > dp_mat[i-1, j]):
                        dp_mat[i, j] = new_value
                    else:
                        dp_mat[i, j] = dp_mat[i-1, j]

                    # Mark best if possible
                    if new_value > best_value:
                        # or (new_value == best_value and i > best_cell[0]):
                        best_value = new_value
                        best_cell = (i, j)

        return dp_mat, best_cell, best_value

    def get_best_clients(dp_mat, cell, value, clients_lst):
        # NOTE: standard DP backtracing algo. Careful with the indexing
        row, col = cell
        w = dp_mat[row, col]
        best_clients = []
        while row > 0 and col > 0:
            if dp_mat[row, col] == dp_mat[row-1, col]:
                row = row - 1
            else:
                client = clients_lst[row - 1]
                w_i = client.fps
                row = row - 1
                col = (w - w_i) // throughput_divisor
                w = col * throughput_divisor
                assert w == dp_mat[row, col]
                best_clients.append(client)

        return best_clients

    def get_best_clients_max_count(dp_mat, cell, value, clients_lst):
        '''
        Function to return client list with maximum cardinality.
        '''
        def get_min_row(row, col):
            min_row = row
            value = dp_mat[row, col]
            min_fps = -1
            while row > 0:
                if value != dp_mat[row, col]:
                    break
                client = clients_lst[row - 1]
                if min_fps == -1 or min_fps > client.fps:
                    # Only valid rows
                    prev_value = value - client.fps
                    prev_row_col = (value - client.fps) // throughput_divisor
                    if prev_row_col >= 0 and dp_mat[row - 1, prev_row_col] == prev_value:
                        min_fps = client.fps
                        min_row = row
                row = row - 1
            return min_row, min_fps

        # NOTE: standard DP backtracking algo. Careful with the indexing
        row, col = cell
        w = dp_mat[row, col]
        best_clients = []
        while row > 0 and col > 0:
            row, _ = get_min_row(row, col)
            client = clients_lst[row - 1]
            w_i = client.fps
            assert(w_i <= w)
            row = row - 1
            col = (w - w_i) // throughput_divisor
            w = col * throughput_divisor
            assert w == dp_mat[row, col]
            best_clients.append(client)

        return best_clients

    # Sort models based on their accuracy and
    # expected it to be sorted also based on latency
    sorted_models = sorted(models_info.values(),
                           key=model_sort_key, reverse=True)
    sorted_clients = clients_info.values()

    # Throughput divisor is a LCD of possible frame rate values
    throughput_divisor = 5
    total_clients = len(sorted_clients)
    models_map = {}
    for model in sorted_models:
        if model.id not in models_map:
            models_map[model.id] = []

    clients_map = {}
    for client in sorted_clients:
        clients_map[client.id] = None

    for model in sorted_models:

        if len(sorted_clients) == 0:
            break

        # Sort clients on the decreasing order of their latency budget for the model
        sorted_clients = sorted(sorted_clients, key=lambda x: client_sort_key(
            x, model.model_number), reverse=True)

        dp_mat, best_cell, best_value = construct_dp_mat(model, sorted_clients)
        if dp_mat is None:
            continue

        best_clients = get_best_clients(
            dp_mat, best_cell, best_value, sorted_clients)
        # best_clients = get_best_clients_max_count(
        #     dp_mat, best_cell, best_value, sorted_clients)

        # For debugging
        total_value = 0
        for client in best_clients:
            total_value += client.fps
        if total_value != best_value:
            assert False, "Total client rate is not equal to the best value in the DP mat"
            # get_best_clients(dp_mat, best_cell, best_value, sorted_clients)

        total_value = 0
        for client in best_clients:
            total_value += client.fps
            assert total_value <= best_value, "Total value is not less than the best value"
            assert clients_map[client.id] is None, "Client already assigned to model"

            clients_map[client.id] = None
            assigned = model.check_and_assign(client)

            assert assigned is True, "Client cannot be assigned"
            clients_map[client.id] = model
            models_map[model.id].append(client)

        sorted_clients = [
            client for client in sorted_clients if client not in best_clients]

    return MappingInfo(models_info, clients_info, models_map, clients_map, name="DP_AggregateRate")
