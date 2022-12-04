import logging
import threading
import time
import numpy as np
from src.server.controller.scheduler import Scheduler
import src.utils as utils
import statistics


''' Control Manager '''


class ControlManager(object):
    def __init__(self, opts, profiled_latencies, profiled_accuracies, worker_manager):
        self._worker_manager = worker_manager
        self._env_data = EnvData(opts, profiled_latencies, profiled_accuracies)
        self._lock = threading.Lock()
        self._close_event = threading.Event()
        self._clients_mapping = None
        self._models_info = None
        self._init_model_info(opts)
        self._init_client_info()
        self._init_scheduler(opts)

    def _init_model_info(self, opts):
        self._model_info = {}
        for gpu_number in range(opts.n_gpus):
            model = DNNModel(
                self._env_data, opts.init_model_number, gpu_number)
            self._model_info[model.id] = model
        self._model_lock = threading.Lock()

    def _init_client_info(self):
        # TODO: Combine clients_info and locks in one collection
        self._client_info = {}
        self._clients_lock = {}
        self._client_seq = 0
        self._client_info_lock = threading.Lock()

    def _init_scheduler(self, opts):
        # Start scheduler process
        self._scheduler = Scheduler(opts, self._env_data, self._model_info)
        self._scheduler.start()

        # Start scheduler interface thread
        self._schedule_min_interval = opts.schedule_min_interval
        self._schedule_interval = max(opts.schedule_interval,
                                     opts.schedule_min_interval)
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_stub)
        self._scheduler_run_event = threading.Event()
        self._scheduler_update_event = threading.Event()
        self._scheduler_thread.start()

        self._scheduler_call_count = 0

    def close(self):
        self._close_event.set()
        self._scheduler.close()
        self._scheduler.join()

    def add_client(self, slo, frame_rate, lat_wire, init_bw):
        logging.info("ControlManager: client add request")
        with self._client_info_lock:
            client_seq = self._client_seq
            self._client_seq += 1
            # Client key is different in mapping algorithms.
            # TODO: make it consistent
            client = Client(env_data=self._env_data, fps=frame_rate,
                            slo=slo, lat_wire=lat_wire, bw=init_bw, id=client_seq)
            client_id = client.id
            self._client_info[client_id] = client
            self._clients_lock[client_id] = threading.Lock()

        # Wait until mapping is available.
        # TODO: Add a timeout, so that client knows that it cannot be mapped
        # and set model_number to -1
        while True:
            gpu_number, model_number, batch_size = self.client_mapping(
                client_seq)
            if gpu_number is not None:
                logging.info(
                    f"Assigning client {client_id} to GPU {gpu_number}")
                break

            self._scheduler_run_event.set()
            self._scheduler_update_event.wait()

        logging.info(f"Client {client_id} mapped to {model_number}")
        return client_seq, model_number

    def remove_client(self, client_seq):
        with self._client_info_lock:
            client_id = Client.getIdString(client_seq)
            del self._client_info[client_id]
            del self._clients_lock[client_id]

    def _get_updated_clients_info(self):
        with self._lock:
            server_overhead = self._env_data.current_server_overhead()
        with self._client_info_lock:
            client_info = self._client_info.copy()
            clients_lock = self._clients_lock
        for client_id, client in client_info.items():
            with clients_lock[client_id]:
                client.save_server_overhead(server_overhead)
                client.update()
        return client_info

    def update_mapping(self, clients_mapping, models_info):
        with self._lock:
            self._clients_mapping = clients_mapping
            self._models_info = models_info

        for _, model in models_info.items():
            logging.info(f"Updating worker on GPU {model.gpu_number} with model"
                         f"number {model.model_number} and batch size {model.batch_count}")
            # Model with batch size 0 will not be used for the infererence.
            # We have two options to do.
            # 1. Keep old model and don't update
            # 2. Use the model returned by the scheduler and set batch size = 1.
            # This option is good, as scheduler will be aware of the model on the GPU.
            # So, shuffling models in the next schedule round, will benefit from the model locality.
            # assert model.batch_count > 0
            batch_size = model.batch_count if model.batch_count else 1
            self._worker_manager.update_worker_model(model.gpu_number,
                                                     model.model_number,
                                                     batch_size)

    def client_mapping(self, client_seq):
        gpu_number, model_number, batch_size = None, None, None
        # TODO: Make client_key and client_id consistent
        client_id = Client.getIdString(client_seq)
        with self._lock:
            if self._clients_mapping is not None and \
                client_id in self._clients_mapping and \
                    self._clients_mapping[client_id] is not None:
                gpu_number = self._clients_mapping[client_id].gpu_number
                model_number = self._clients_mapping[client_id].model_number
                batch_size = self._clients_mapping[client_id].batch_count
        return gpu_number, model_number, batch_size

    def clients_next_model(self, client_seq):
        def _backoff(model_number):
            # _POLICY = "AGGRESIVE_BACKOFF"
            _POLICY = "BEST_BACKOFF"
            if _POLICY == "AGGRESIVE_BACKOFF":
                return 0
            elif _POLICY == "BEST_BACKOFF":
                return model_number - 1
            else:
                return model_number

        gpu, next_model, batch_size = self.client_mapping(client_seq)
        if next_model is None:
            return -1

        assert batch_size is not None
        '''
        Ideally, we should send the model returned by the clients mapping.
        However, as scheduler might still be running to handle the change in the clients network
        We should send client the next best model that clients network can support.
        This will lead to a mismatch between the model on the GPU and frame size. However, this will
        only lead to an accuracy degradation for a short while.
        '''
        model_number = next_model
        client, client_lock = self._get_client(client_seq)
        with client_lock:
            client_bw = client.bw
            client_fps = client.fps
            client_lat_wire = client.lat_wire
            client_response_size = client.response_size
            client_slo = client.slo

        while model_number > 0:
            with self._lock:
                latency_model = self._env_data.lat_m[next_model][batch_size - 1]
                server_overhead = self._env_data.current_server_overhead()

            # First check for the throughput
            net_throughput = (client_bw / self._env_data.frame_s[model_number])
            if net_throughput <= client_fps:
                model_number = _backoff(model_number)
                continue

            # Now check for the latency budget
            data_size = self._env_data.frame_s[model_number] + \
                client_response_size
            lat_net = utils.compute_net_lat(
                data_size, client_bw, client_lat_wire)
            budget = client_slo - lat_net - server_overhead
            if budget < (2 * latency_model):
                model_number = _backoff(model_number)
                continue

            break

        logging.debug(
            f"Clients {client_seq} next model: model on GPU {next_model} and model size on network {model_number}")

        # TODO: Check if the desired next_model does not match with the current model_number.
        # Maintain an array of mismatch between next_model and model_number, and if the state has changed
        # too drastically then run the scheduler if the frequency of scheduler run is very low.

        next_model = model_number
        return next_model

    def _scheduler_stub(self):
        """
            Scheduler stub/interface to the Scheduler process.
            It runs periodically and when control manager asks for.
            Avoid running the scheduler too frequently due to the cost of switching models.
            Keep a minimum interval between two consecutive runs.
        """
        def wait_for_min_interval(last_run_ts, min_interval):
            elapsed_time = time.time() - last_run_ts
            wait_time_in_ms = (min_interval - elapsed_time) * 1e3
            if wait_time_in_ms > 0:
                utils.sleepInMillis(wait_time_in_ms)
            return

        logging.info("Started scheduler stub!")
        last_run_ts = time.time() - self._schedule_min_interval
        while not self._close_event.is_set():
            # Wait for the scheduler run event, either for
            # 1. Periodic interval timeout
            # 2. Trigger when new clients join or leave
            # 3. Trigger when clients bw changes drastically. @TODO: Implement this.
            self._scheduler_run_event.wait(timeout=self._schedule_interval)

            # Wait for atleast minimum interval before the next call to the scheduler.
            # To prevent frequent updates.
            wait_for_min_interval(last_run_ts, self._schedule_min_interval)

            # Start the scheduler run
            client_info = self._get_updated_clients_info()
            self._scheduler_update_event.clear()
            if len(client_info) > 0:
                self._scheduler_call_count += 1  # TODO: this need lock protection
                logging.debug(f"Request scheduler to map clients,"
                              f" count {self._scheduler_call_count}")
                clients_map, models_info = self._scheduler.map_clients(
                    client_info)
                self.update_mapping(clients_map, models_info)
                last_run_ts = time.time()
                logging.debug(f"Scheduler done mapping, count"
                              f" {self._scheduler_call_count}")

            self._scheduler_update_event.set()
            self._scheduler_run_event.clear()

    def _run_scheduler_for_bw_change(self, prev_bw, curr_bw):
        _DELTA_BW = 2.5 * 1024  # kbps
        if abs(curr_bw - prev_bw) >= _DELTA_BW:
            self._scheduler_run_event.set()
            logging.debug(f"Scheduler run triggered due to bw change:"
                          f" Previous {prev_bw}, Current {curr_bw}")

    def _get_client(self, client_seq):
        client_id = Client.getIdString(client_seq)
        with self._client_info_lock:
            client = self._client_info[client_id]
            client_lock = self._clients_lock[client_id]
        return client, client_lock

    def update_request_metadata(self, client_seq, metadata):
        client, client_lock = self._get_client(client_seq)

        data_size = utils.bytes_to_kbits(metadata.request_size)
        with client_lock:
            data_size += client.response_size
            lat_wire = client.lat_wire
            metadata.slo = client.slo
        metadata.time_on_network = utils.compute_net_lat(
            data_size, metadata.client_bw, lat_wire)

    def save_request_stats(self, client_seq, request_size, client_bw, metadata):
        client, client_lock = self._get_client(client_seq)
        with client_lock:
            net_time = (metadata.server_recv_ts -
                        metadata.client_send_ts) * 1e3
            logging.debug(
                f"Timestamp: {metadata.client_send_ts}->{metadata.server_recv_ts} frame id {metadata.frame_id}")
            prev_bw = client.bw
            client.save_bw(metadata.desired_model,
                           request_size, net_time, client_bw)
            curr_bw = client.bw

        # TODO: Check if we need to run the scheduler due to the bw change
        # self._run_scheduler_for_bw_change(prev_bw, curr_bw)

    def save_response_stats(self, client_seq, response_size, metadata):
        client, client_lock = self._get_client(client_seq)
        dropped_frame_flag = (metadata.used_model < 0)
        with client_lock:
            client_bw = client.bw
            if not dropped_frame_flag:
                # Not a dropped frame, so contains correct response
                client.save_response_size(response_size)

        # Save extra metadata per frame
        metadata.estimated_bw = client_bw
        metadata.scheduler_run_count = self._scheduler_call_count
        metadata.response_size = response_size

        # Save server overhead
        if not dropped_frame_flag:
            dispatch_time = (metadata.worker_recv_ts -
                             metadata.server_recv_ts) * 1e3
            processing_time = (metadata.worker_processing_ts -
                               metadata.worker_recv_ts) * 1e3
            with self._lock:
                self._env_data.update_server_overhead(
                    dispatch_time, processing_time)


class EnvData(object):
    def __init__(self, opts, profiled_latencies, profiled_accuracies):
        self.num_models = opts.n_models
        self.max_batch_size = opts.max_batch_size
        self.batch_size = [i for i in range(1, opts.max_batch_size+1)]
        self.num_gpus = opts.n_gpus
        # Hard coded fps_lcd at this point. It depends on the clients' frame rates
        self.fps_lcd = opts.fps_lcd
        self.effectiveness_threshold = opts.effectiveness_threshold
        self.acc_m = np.array(profiled_accuracies)
        self.lat_m = np.array(profiled_latencies)

        # Compute the average or max frame size over the wire, to compute network latency
        self.frame_s = utils.compute_frame_size(self.num_models)
        # Compute model throughput
        throughput_m = utils.compute_model_throughput(self.lat_m)
        self.throughput_m = np.asarray(throughput_m)

        # Variables that changes due to system dynamics.
        self.dispatch_time = 1
        self.preprocessing_time = 0
        # This is the extra time spent on the server, for example,
        # 1. Executor checking if new model is available, timing calls etc.
        self.other_noise_time = 2  # conservative

    def update_server_overhead(self, dispatch_time, preprocessing_time, other_noise_time=2):
        self.dispatch_time = 0.5 * self.dispatch_time + 0.5 * dispatch_time
        self.preprocessing_time = 0.5 * self.preprocessing_time + 0.5 * preprocessing_time
        self.other_noise_time = 0.5 * self.other_noise_time + 0.5 * other_noise_time

    def current_server_overhead(self):
        server_overhead = 2*self.dispatch_time + \
            self.preprocessing_time + self.other_noise_time

        return server_overhead

    def print_data(self):
        print("Model Accuracy():", self.acc_m)
        print("Model Latency(ms), Batch Size(1):", self.lat_m[:, 0])
        print("Model Throughput(fps), Batch Size(1):", self.throughput_m[:, 0])


''' Define classes for DNNModels and state manager '''


class Client:
    def __init__(self, env_data, fps, slo, lat_wire, bw, id):

        self.env_data = env_data
        self.client_number = id
        self.id = self.getIdString(id)
        self.fps = fps
        self.slo = slo
        self.bw = bw
        self.lat_wire = lat_wire
        self.lat_net = np.zeros(env_data.num_models, dtype=float)
        self.lat_budget = np.zeros(env_data.num_models, dtype=float)
        self.throughput_net = np.zeros(env_data.num_models, dtype=float)

        self._bw_history = [bw]
        self.response_size = utils.bytes_to_kbits(300)
        self._server_overhead = 2*1+0+2  # 2 * dispatch + preprocessing + other_noise

        self.update()

    @staticmethod
    def getIdString(id):
        return "client_" + str(id)

    def reset(self):
        self.lat_net = np.zeros(self.env_data.num_models, dtype=float)
        self.lat_budget = np.zeros(self.env_data.num_models, dtype=float)
        self.throughput_net = np.zeros(self.env_data.num_models, dtype=float)

    def save_bw(self, model_number, request_size, net_time, client_bw):
        _HISTORY_INTERVAL = 1  # in seconds
        if len(self._bw_history) == (self.fps * _HISTORY_INTERVAL):
            del self._bw_history[0]

        self._bw_history.append(client_bw)
        self.bw = self.estimate_bw()

    def save_response_size(self, response_size):
        # May be we don't need to consider the response_size,
        # as it is has a negligle impact on the network latency.
        self.response_size = utils.bytes_to_kbits(response_size)

    def estimate_bw(self):
        # Harmonic mean of past network throughputs
        # TODO: We can also use moving average or smoothing filters
        bw = statistics.harmonic_mean(self._bw_history)
        logging.info(f"Client {self.id} estimated bw is {bw}")
        return bw

    def save_server_overhead(self, server_overhead):
        self._server_overhead = server_overhead

    def update(self):
        self.bw = self.estimate_bw()
        self._update_lat_net()
        self._update_throughput_net()
        self._update_lat_budget()
        # self._update_lat_budget_fixed_slo() # Use for data adaptation with slo_awareness
        # self._update_lat_budget_fixed_frame() # Use for data adaptation with fixed frame size

    def _update_lat_net(self):
        for i in range(self.env_data.num_models):
            data_size = self.env_data.frame_s[i] + self.response_size
            lat = utils.compute_net_lat(data_size, self.bw, self.lat_wire)
            self.lat_net[i] = lat

    def _update_lat_budget_fixed_slo(self):
        # Consider latency budget as some percentage of e2e latency SLO
        _PERCENTAGE = 0.75
        for i in range(self.env_data.num_models):
            self.lat_budget[i] = self.slo * _PERCENTAGE

    def _update_lat_budget_fixed_frame(self):
        # Consider fixed frame size sent by the client.
        # _CLIENT_FRAME_SIZE = self.env_data.num_models - 1
        _CLIENT_FRAME_SIZE = 7
        fixed_budget = self.slo - \
            self.lat_net[_CLIENT_FRAME_SIZE] - self._server_overhead
        for i in range(self.env_data.num_models):
            self.lat_budget[i] = max(0, fixed_budget)

    def _update_lat_budget(self):
        for i in range(self.env_data.num_models):
            budget = self.slo - self.lat_net[i] - self._server_overhead

            if self.throughput_net[i] > self.fps:
                self.lat_budget[i] = max(0, budget)
            else:
                self.lat_budget[i] = 0.0

    def _update_throughput_net(self):
        for i in range(self.env_data.num_models):
            self.throughput_net[i] = (self.bw / self.env_data.frame_s[i])


class DNNModel:
    def __init__(self, env_data, model_number, gpu_number):
        self.id = self.getIdString(model_number, gpu_number)
        self.model_number = model_number
        self.gpu_number = gpu_number
        self.max_batch_size = len(env_data.batch_size)
        self.accuracy = env_data.acc_m[model_number]
        self.latency_vec = env_data.lat_m[model_number]
        self.throughput_vec = env_data.throughput_m[model_number]
        self.assigned_clients = []
        self.batch_count = 0
        self.lambda_j = 0

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
            logging.warn("DNNModel: Batch count is lower")
            return False

        if check_throughput and not self._check_throughput_constraint(total_rate, batch_count):
            logging.warn("DNNModel: Throughput constaint failed")
            return False

        if check_latency and not self._check_latency_constraint(new_client, batch_count):
            logging.warn("DNNModel: Latency constraint failed")
            return False

        # Increase batch count for larger batch size preference
        for next_batch_count in range(batch_count + 1, self.max_batch_size):
            if (self._check_throughput_constraint(total_rate, next_batch_count) and
                    self._check_latency_constraint(new_client, next_batch_count)):
                batch_count = next_batch_count
            else:
                break

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
