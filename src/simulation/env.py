import numpy as np
import random
import math
import src.utils as utils


### Data generators ###


def generate_slo():
    _SLO = [75, 100, 150]
    idx = random.randint(0, len(_SLO)-1)
    return _SLO[idx]


def generate_fps(fps_lcd=5):
    _FPS = [10, 15, 25]
    idx = random.randint(0, len(_FPS)-1)
    return _FPS[idx]


def generate_bw():
    # TODO: Read bw trace from files
    def uniform_bw():
        _BW_LO = 7.5
        _BW_HIGH = 50
        bw = random.uniform(_BW_LO, _BW_HIGH) * 1024
        return bw

    return uniform_bw()


def generate_frame_resolution_lst(num_model=None):
    def _resolution_to_size(frame_r):
        _COMPRESSION_RATIO = 8
        frame_s = [(frame_r[i] * frame_r[i] * 24) /
                   (1024 * _COMPRESSION_RATIO) for i in range(len(frame_r))]
        return frame_s

    _RESOLUTION = [256 + 64 * i for i in range(1, 20)]
    _SIZE = _resolution_to_size(_RESOLUTION)

    if num_model is None:
        return _RESOLUTION, _SIZE
    else:
        idx = np.linspace(0, len(_RESOLUTION) - 1, num_model, dtype=int)
        res = [_RESOLUTION[i] for i in idx]
        size = [_SIZE[i] for i in idx]
        return res, size


def generate_model_accuracy_profiles(frame_r):
    def _get_function():  # Functions
        alpha_1, alpha_2, alpha_3 = 0.988, 4.469, -1/200  # [JOINT, INFOCOM'20]
        # alpha_1, alpha_2, alpha_3 = 1.091, 0.6166, -1/0.5218 # [IEEE ACCESS'20]
        def func_acc(r): return alpha_1 - alpha_2 * math.exp(r * alpha_3)
        return func_acc

    func_acc = _get_function()
    acc_m = [func_acc(frame_r[i]) for i in range(len(frame_r))]
    return acc_m


def generate_model_latency_profiles(frame_r, num_models, batch_size):
    def _get_functions():
        def func_batch_lat(first_lat):
            '''
            Latency function w.r.t batch size
            '''
            batch_latency = []
            _SPLIT_FACTOR = 0.75
            alpha = _SPLIT_FACTOR * first_lat
            beta = (1-_SPLIT_FACTOR) * first_lat
            # Generate latecy values
            for i in range(len(batch_size)):
                lat = alpha + beta * i
                batch_latency.append(lat)
            return batch_latency

        def func_lat(r):
            '''
            Latency function w.r.t frame resolution
            '''
            beta_1, beta_2, beta_3 = 0.08315, -0.04463, 0.03757
            return (beta_1 * r * r + beta_2 * r + beta_3) / 1000.0
        return func_lat, func_batch_lat

    func_lat, func_batch_lat = _get_functions()
    lat_m = [func_batch_lat(func_lat(frame_r[i])) for i in range(num_models)]
    return lat_m


def profiled_frame_resolution_lst(num_models):
    return utils.FRAME_SIZES, utils.compute_frame_size(num_models)


class SimData(object):
    '''
    Class to hold the simulation data
    '''

    def __init__(self, opts):
        self.num_clients = opts.num_clients
        self.num_models = opts.num_models
        self.max_batch_size = opts.max_batch_size
        self.batch_size = [i for i in range(1, opts.max_batch_size+1)]
        self.num_gpus = opts.num_gpus
        self.fps_lcd = opts.fps_lcd
        self.effectiveness_threshold = opts.effectiveness_threshold

        self.generate_data(opts)

    def generate_data(self, opts):
        # Client attributes
        fps = [generate_fps(self.fps_lcd) for i in range(self.num_clients)]
        slo = [generate_slo() for i in range(self.num_clients)]
        bw = [generate_bw() for i in range(self.num_clients)]  # Kbps

        # Generate or retrieve model profiles
        if not opts.use_profiled_values:
            frame_r, frame_s = profiled_frame_resolution_lst(
                num_models=self.num_models)
            acc_m = generate_model_accuracy_profiles(frame_r)
            lat_m = generate_model_latency_profiles(frame_r,
                                                    self.num_models,
                                                    self.batch_size)
            throughput_m = utils.compute_model_throughput(lat_m)
        else:
            frame_r, frame_s = profiled_frame_resolution_lst(
                num_models=self.num_models)
            acc_m = utils.read_profiled_accuracies(opts.profiled_dir,
                                                   self.num_models)
            acc_m = acc_m
            lat_m = utils.read_profiled_latencies(opts.profiled_dir,
                                                  self.num_models,
                                                  self.max_batch_size)
            _LAT_SCALE = 1.00
            lat_m = lat_m * _LAT_SCALE
            throughput_m = utils.compute_model_throughput(lat_m)

        # convert data to numpy arrays
        self.fps = np.asarray(fps)
        self.slo = np.asarray(slo)
        self.bw = np.asarray(bw)
        self.frame_r = np.asarray(frame_r)
        self.frame_s = np.asarray(frame_s)
        self.acc_m = np.asarray(acc_m)
        self.lat_m = np.asarray(lat_m)
        self.throughput_m = np.asarray(throughput_m)

        return

    def print_data(self):
        print("Client SLOs(ms):", self.slo)
        print("Client BW(kbps):", self.bw)
        print("Model frame resolutions:", self.f_res)
        print("Model frame sizes(kb):", self.f_size)
        print("Model accuracies():", self.acc_m)
        print("Model latencies(ms) for batch size (1):", self.lat_m[:, 0])
        print("Model throughput(fps) for batch Size (1):",
              self.throughput_m[:, 0])
        print("Network latencies(ms):", self.lat_net[:, 0])


### Helper functions ###


def print_info(clients_info, models_info):

    print('{:=^100}'.format('Model/Client Info'))
    for _, model in models_info.items():
        print("Model Id:", model.id)
        print("\t" + "-"*60)
        print("\t{:<20s}{:<20s}{:<20s}".format("BatchCount",
                                               "ModelLatency",
                                               "ModelThroughput"))
        print("\t" + "-"*60)
        for i in range(len(model.latency_vec)):
            print("\t{:<20d}{:<20.2f}{:<20.2f}".format(i+1,
                                                       model.latency_vec[i],
                                                       model.throughput_vec[i]))

    for _, client in clients_info.items():
        print("Client Id:", client.id)
        print("\tFPS:", client.fps)
        print("\tSLO:", client.slo)
        print("\t" + "-"*80)
        print("\t{:<20s}{:<20s}{:<20s}{:<20s}".format("ModelNumber",
                                                      "LatencyNet",
                                                      "ThroughputNet",
                                                      "LatencyBudget"))
        print("\t" + "-"*80)
        for i in range(len(client.lat_net)):
            print("\t{:<20d}{:<20.2f}{:<20.2f}{:<20.2f}".format(i,
                                                                client.lat_net[i],
                                                                client.throughput_net[i],
                                                                client.lat_budget[i]))

    print('{:=^100}'.format(""))


### Define client and model classes ###


class Client:
    '''
    Class to represent a client
    '''

    def __init__(self, sim_data, id):

        self.sim_data = sim_data
        self.client_number = id
        self.id = "client_" + str(id)
        self.fps = sim_data.fps[id]
        self.slo = sim_data.slo[id]
        self.bw = sim_data.bw[id]
        self.lat_net = np.zeros(sim_data.num_models, dtype=float)
        self.lat_budget = np.zeros(sim_data.num_models, dtype=float)
        self.throughput_net = np.zeros(sim_data.num_models, dtype=float)

        self.next()

    def reset(self):
        self.lat_net = np.zeros(self.sim_data.num_models, dtype=float)
        self.lat_budget = np.zeros(self.sim_data.num_models, dtype=float)
        self.throughput_net = np.zeros(self.sim_data.num_models, dtype=float)

    def next(self):
        self.bw = self.bw
        self.update()

    def update(self):
        self._update_lat_net()
        self._update_throughput_net()
        self._update_lat_budget()

    def _update_lat_net(self):
        for i in range(self.sim_data.num_models):
            lat = utils.compute_net_lat(self.sim_data.frame_s[i], self.bw)
            self.lat_net[i] = lat

    def _update_lat_budget(self):
        for i in range(self.sim_data.num_models):
            budget = self.slo - self.lat_net[i]
            if self.throughput_net[i] > self.fps:
                self.lat_budget[i] = max(0, budget)
            else:
                self.lat_budget[i] = 0.0

    def _update_throughput_net(self):
        for i in range(self.sim_data.num_models):
            self.throughput_net[i] = (self.bw / self.sim_data.frame_s[i])


class DNNModel:
    '''
    Class to represent a DNN model
    '''

    def __init__(self, sim_data, model_number, gpu_number):
        self.id = self.getIdString(model_number, gpu_number)
        self.model_number = model_number
        self.gpu_number = gpu_number
        self.max_batch_size = len(sim_data.batch_size)
        self.accuracy = sim_data.acc_m[model_number]
        self.latency_vec = sim_data.lat_m[model_number]
        self.throughput_vec = sim_data.throughput_m[model_number]
        self.assigned_clients = []  # clients assigned to this model
        self.batch_count = 0        # batch size to use with this model
        self.lambda_j = 0           # aggregate request rates

    def reset(self):
        self.assigned_clients = []
        self.batch_count = 0
        self.lambda_j = 0

    def changeGPU(self, gpu_number):
        self.id = self.getIdString(self.model_number, gpu_number)
        self.gpu_number = gpu_number

    @staticmethod
    def getIdString(id, sub_id):
        return "model_" + str(id) + "_" + str(sub_id)

    def check_and_assign(self, new_client, check_throughput=True, check_latency=True):
        total_rate = 0
        for client in self.assigned_clients:
            assert client.id != new_client.id, "Assigning the same client"
            total_rate += client.fps

        total_rate += new_client.fps
        batch_count = self._get_batch_count(total_rate)
        if batch_count < 1:
            return False

        if check_throughput and not self._check_throughput_constraint(total_rate, batch_count):
            return False

        if check_latency and not self._check_latency_constraint(new_client, batch_count):
            return False

        # @NOTE: check the control manager in the runtime code to see that we can try to
        # to increase the batch count without hurting latency contraints. This is for
        # larger batch size preference.
        self.batch_count = batch_count
        self.assigned_clients.append(new_client)
        self.lambda_j = total_rate
        return True

    def _get_batch_count(self, total_rate):
        batch_count = -1
        for idx in range(self.max_batch_size):
            if (total_rate < self.throughput_vec[idx]):
                batch_count = idx + 1
                break
        return batch_count

    def _check_throughput_constraint(self, total_rate, batch_count):
        if total_rate < self.throughput_vec[batch_count-1]:
            return True
        else:
            return False

    def _check_latency_constraint(self, new_client, batch_count):
        latency_model = self.latency_vec[batch_count-1]
        if (new_client.lat_budget[self.model_number] < 2 * latency_model):
            return False
        for client in self.assigned_clients:
            if (client.lat_budget[self.model_number] < 2 * latency_model):
                return False
        return True

### Define classes for mapping info and metric ###


class MappingMetrics(object):
    '''
    Class to compute and print metrics for clients mapped to DNNs
    '''

    def __init__(self, models_info, clients_info, models_map, clients_map) -> None:

        self.client_metrics(clients_info, clients_map)
        self.model_metrics(models_info, models_map)

    @staticmethod
    def effective_client(client):
        smallest_model = -1
        if client.lat_budget[smallest_model+1]:
            return True
        return False

    @staticmethod
    def get_slack_residue(model, client):
        batch_count = model.batch_count
        lat_model = model.latency_vec[batch_count-1]
        residue = (client.lat_budget[model.model_number] - 2 * lat_model)
        return residue

    def _accuracy_metrics(self, accuracy_dict, clients_map):
        assert len(accuracy_dict) > 0, "Accuracy dict is empty"
        acc_per_client = sum(accuracy_dict.values()) / \
            float(len(accuracy_dict))
        acc_per_requests = 0.0
        total_requests = 0
        for client_id, acc in accuracy_dict.items():
            requests = clients_map[client_id].fps
            total_requests += requests
            acc_per_requests += acc * requests

        acc_per_requests /= total_requests
        return acc_per_client, acc_per_requests

    def client_metrics(self, clients_info, clients_map):
        assert len(clients_info) == len(
            clients_map), "Client mapping length mismatch"
        rejected_effective = 0
        total_effective = 0
        self.slack_residue = {}
        # accuracy of selected clients. Not an effective accuracy.
        # Need to check SPINN or DDS for effective accuracy that multiplies.
        # Number of effective clients with the accuracy.
        accuracy = {}
        total_effective_rate = 0
        total_rejected_rate = 0
        for client_id, model in clients_map.items():
            client = clients_info[client_id]
            is_effective_client = self.effective_client(client)
            if is_effective_client:
                total_effective += 1
                total_effective_rate += client.fps
                if model is not None:
                    self.slack_residue[client_id] = self.get_slack_residue(model,
                                                                           client)
                    accuracy[client_id] = model.accuracy
                else:
                    rejected_effective += 1
                    total_rejected_rate += client.fps
                    accuracy[client_id] = 0.0
            else:
                assert model is None, print(
                    "Model is not none for ineffective client")

        self.effectiveness = 0.0
        self.accuracy_per_request = 0.0
        self.accuracy_per_client = 0.0
        if total_effective > 0:
            self.effectiveness = 1 - rejected_effective / total_effective
            # self.effectiveness = 1 - total_rejected_rate / total_effective_rate
        if self.effectiveness > 0.0:
            self.accuracy_per_client, self.accuracy_per_request = self._accuracy_metrics(accuracy,
                                                                                         clients_info)

    def model_metrics(self, models_info, models_map):
        assert len(models_info) == len(
            models_map), "Model mapping length mistmatch"
        self.utilization = {}
        for model_id, clients in models_map.items():
            model = models_info[model_id]
            batch_count = model.batch_count
            lambda_j = 0
            for client in clients:
                lambda_j += client.fps
            assert (model.lambda_j ==
                    lambda_j), "MappingMetrics: Model lambda is not equal to total client rate"
            self.utilization[model_id] = lambda_j / \
                model.throughput_vec[batch_count - 1]

        self.utilization_avg = 0.0
        if len(self.utilization) > 0:
            self.utilization_avg = sum(
                self.utilization.values()) / len(models_map)

    def print(self):
        # Overall metrics
        print("\t" + "-"*80)
        print("\t{:<20s}{:<20s}{:<20s}{:<20s}".format("Effectiveness(%)",
                                                      "AccuracyPerRequest",
                                                      "AccuracyPerClient",
                                                      "Utilization"))
        print("\t" + "-"*80)
        print("\t{:<20.2f}{:<20.2f}{:<20.2f}{:<20.2f}".format(self.effectiveness,
                                                              self.accuracy_per_request,
                                                              self.accuracy_per_client,
                                                              self.utilization_avg))

        # Detailed metrics
        print("\t" + "-"*40)
        print("\t{:<20s}{:<20s}".format("Clients", "SlackResidue(ms)"))
        print("\t" + "-"*40)
        for key in sorted(self.slack_residue):
            print("\t{:<20s}{:<20.2f}".format(
                key, self.slack_residue[key]))

        print("\t" + "-"*40)
        print("\t{:<20s}{:<20s}".format("Models", "Utilization(%)"))
        print("\t" + "-"*40)
        for key in sorted(self.utilization):
            print("\t{:<20s}{:<20.2f}".format(
                key, self.utilization[key]))

        print('{:=^100}'.format(""))


class MappingInfo(object):
    '''
    Class to store and print mapping information
    '''

    def __init__(self, models_info, clients_info, models_map, clients_map, name=""):
        self.models_info = models_info
        self.clients_info = clients_info
        self.models_map = models_map
        self.clients_map = clients_map
        self.name = name

        self.metrics = MappingMetrics(models_info,
                                      clients_info,
                                      models_map,
                                      clients_map)

    def print(self):
        print('{:=^100}'.format('Mapping Info: '+self.name))
        for model_id, clients in self.models_map.items():
            print("Model:\t{}".format(model_id))
            model = self.models_info[model_id]
            print("\t" + "-"*60)
            print("\t{:<10s}{:<5s}{:<5s}{:<20s}{:<20s}".format("Client",
                                                               "FPS",
                                                               "SLO",
                                                               "LatencyNetModel",
                                                               "LatencyBudgetModel"))
            print("\t" + "-"*60)
            sorted_clients = sorted(clients, key=lambda x: x.id)
            lambda_j = 0
            for client in sorted_clients:
                assert(model.id == self.clients_map[client.id].id)
                print("\t{:<10s}{:<5d}{:<5d}{:<20.2f}{:<20.2f}".format(client.id,
                                                                       client.fps,
                                                                       client.slo,
                                                                       client.lat_net[model.model_number],
                                                                       client.lat_budget[model.model_number]))

                lambda_j += client.fps

            assert (model.lambda_j ==
                    lambda_j), "ModelInfo: Model lambda does not match with total client rate"
            print("\n")
            batch_count = model.batch_count

            print("\t" + "-"*70)
            print("\t{:<15s}{:<15s}{:<20s}{:<20s}".format("BatchCount",
                                                          "Lambda",
                                                          "ModelLatencyBatch",
                                                          "ModelThroughputBatch"))
            print("\t" + "-"*70)
            if batch_count:
                print("\t{:<15d}{:<15d}{:<20.2f}{:<20.2f}".format(batch_count,
                                                                  model.lambda_j,
                                                                  model.latency_vec[batch_count-1],
                                                                  model.throughput_vec[batch_count-1]))

        print('{:=^100}'.format(""))
