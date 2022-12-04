
def printModelIDsOnGpu(modelsInfo):
    sorted_model_ids = [model_id for model_id, model in sorted(
        modelsInfo.items(), reverse=False, key=lambda x: x[1].gpu_number)]
    return sorted_model_ids


def print_info(clients_info, models_info):
    return print(print_info_str(clients_info, models_info))


def print_info_str(clients_info, models_info):

    fmt_str = "\n" + '{:=^100}'.format('Model/Client Info') + "\n"
    for _, model in models_info.items():
        fmt_str += f"Model Id:\t{model.id}" + "\n"
        fmt_str += "\t" + "-"*60 + "\n"
        fmt_str += "\t{:<20s}{:<20s}{:<20s}".format(
            "BatchCount", "ModelLatency", "ModelThroughput") + "\n"
        fmt_str += "\t" + "-"*60 + "\n"
        for i in range(len(model.latency_vec)):
            fmt_str += "\t{:<20d}{:<20.2f}{:<20.2f}".format(
                i+1, model.latency_vec[i], model.throughput_vec[i]) + "\n"

    for _, client in clients_info.items():
        fmt_str += f"Client Id:\t{client.id}" + "\n"
        fmt_str += f"\tFPS:\t{client.fps}" + "\n"
        fmt_str += f"\tSLO:\t{client.slo}" + "\n"
        fmt_str += f"\tBW:\t{client.bw}" + "\n"
        fmt_str += "\t" + "-"*80 + "\n"
        fmt_str += "\t{:<20s}{:<20s}{:<20s}{:<20s}".format(
            "ModelNumber", "LatencyNet", "ThroughputNet", "LatencyBudget") + "\n"
        fmt_str += "\t" + "-"*80 + "\n"
        for i in range(len(client.lat_net)):
            fmt_str += "\t{:<20d}{:<20.2f}{:<20.2f}{:<20.2f}".format(
                i, client.lat_net[i], client.throughput_net[i], client.lat_budget[i]) + "\n"

    fmt_str += '{:=^100}'.format("") + "\n"
    return fmt_str

# Define classes for mapping info and metric


class MappingMetrics(object):
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
                    self.slack_residue[client_id] = self.get_slack_residue(
                        model, client)
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
            self.effectiveness = 1 - rejected_effective/total_effective
            # self.effectiveness = 1 - total_rejected_rate / total_effective_rate
        if self.effectiveness > 0.0:
            self.accuracy_per_client, self.accuracy_per_request = self._accuracy_metrics(
                accuracy, clients_info)

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

    def print_str(self):
        # Overall metrics
        fmt_str = "\n"
        fmt_str += "\t" + "-"*80 + "\n"
        fmt_str += "\t{:<20s}{:<20s}{:<20s}{:<20s}".format(
            "Effectiveness(%)", "AccuracyPerRequest", "AccuracyPerClient", "Utilization") + "\n"
        fmt_str += "\t" + "-"*80 + "\n"
        fmt_str += "\t{:<20.2f}{:<20.2f}{:<20.2f}{:<20.2f}".format(self.effectiveness,
                                                                   self.accuracy_per_request,
                                                                   self.accuracy_per_client,
                                                                   self.utilization_avg) + "\n"

        # Detailed metrics
        fmt_str += "\t" + "-"*40 + "\n"
        fmt_str += "\t{:<20s}{:<20s}".format("Clients",
                                             "SlackResidue(ms)") + "\n"
        fmt_str += "\t" + "-"*40 + "\n"
        for key in sorted(self.slack_residue):
            fmt_str += "\t{:<20s}{:<20.2f}".format(
                key, self.slack_residue[key]) + "\n"

        fmt_str += "\t" + "-"*40 + "\n"
        fmt_str += "\t{:<20s}{:<20s}".format("Models", "Utilization(%)") + "\n"
        fmt_str += "\t" + "-"*40 + "\n"
        for key in sorted(self.utilization):
            fmt_str += "\t{:<20s}{:<20.2f}".format(
                key, self.utilization[key]) + "\n"

        fmt_str += '{:=^100}'.format("") + "\n"
        return fmt_str


class MappingInfo(object):
    def __init__(self, models_info, clients_info, models_map, clients_map, name=""):
        self.models_info = models_info
        self.clients_info = clients_info
        self.models_map = models_map
        self.clients_map = clients_map
        self.name = name

        self.metrics = MappingMetrics(
            models_info, clients_info, models_map, clients_map)

    def print_str(self):
        fmt_str = "\n"
        fmt_str += '{:=^100}'.format('Mapping Info: '+self.name) + "\n"
        for model_id, clients in self.models_map.items():
            fmt_str += "Model:\t{}".format(model_id) + "\n"
            model = self.models_info[model_id]
            fmt_str += "\t" + "-"*60 + "\n"
            fmt_str += "\t{:<10s}{:<5s}{:<5s}{:<20s}{:<20s}".format(
                "Client", "FPS", "SLO", "LatencyNetModel", "LatencyBudgetModel") + "\n"
            fmt_str += "\t" + "-"*60 + "\n"
            sorted_clients = sorted(clients, key=lambda x: x.id)
            lambda_j = 0
            for client in sorted_clients:
                assert(model.id == self.clients_map[client.id].id)
                fmt_str += "\t{:<10s}{:<5d}{:<5d}{:<20.2f}{:<20.2f}".format(client.id,
                                                                            client.fps,
                                                                            client.slo,
                                                                            client.lat_net[model.model_number],
                                                                            client.lat_budget[model.model_number]) + "\n"

                lambda_j += client.fps

            assert (model.lambda_j ==
                    lambda_j), "ModelInfo: Model lambda does not match with total client rate"
            fmt_str += "\n"
            batch_count = model.batch_count

            fmt_str += "\t" + "-"*70 + "\n"
            fmt_str += "\t{:<15s}{:<15s}{:<20s}{:<20s}".format(
                "BatchCount", "Lambda", "ModelLatencyBatch", "ModelThroughputBatch") + "\n"
            fmt_str += "\t" + "-"*70 + "\n"
            if batch_count:
                fmt_str += "\t{:<15d}{:<15d}{:<20.2f}{:<20.2f}".format(batch_count, model.lambda_j,
                                                                       model.latency_vec[batch_count-1],
                                                                       model.throughput_vec[batch_count-1]) + "\n"

        fmt_str += '{:=^100}'.format("") + "\n"

        return fmt_str
