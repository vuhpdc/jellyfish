import logging
from src.utils import Logger
import threading
import queue
import time
import os


class RequestDispatcher(object):
    def __init__(self, opts, worker_manager, control_manager) -> None:
        self._worker_manager = worker_manager
        self._control_manager = control_manager
        self._lock = threading.Lock()
        self._client_cvs_dict = {}
        self._client_queues_dict = {}
        self._close_event = threading.Event()
        self._response_handler_thread = threading.Thread(
            target=self._response_handler)
        self._response_handler_thread.start()

        # stats
        self._stats_logger = Logger(os.path.join(opts.log_path, "frame_path.csv"),
                                    ['client_id', 'frame_id', 'send_net_time',
                                     'dispatch_time', 'processing_time', 'waiting_time',
                                     'prediction_time', 'client_bw', 'estimated_bw', 'request_size',
                                     'response_size', 'scheduler_run_count',
                                     'used_model', 'gpu_number'])

    def close(self):
        self._close_event.set()
        self._response_handler_thread.join()

    def register_client(self, slo, frame_rate, lat_wire, init_bw):
        client_id, model_number = self._control_manager.add_client(
            slo, frame_rate, lat_wire, init_bw)
        with self._lock:
            self._client_cvs_dict[client_id] = threading.Condition()
            self._client_queues_dict[client_id] = queue.Queue()
        return client_id, model_number

    def unregister_client(self, client_id):
        self._control_manager.remove_client(client_id)
        with self._lock:
            del self._client_cvs_dict[client_id]
            del self._client_queues_dict[client_id]

    def _get_client_lock(self, client_id):
        with self._lock:
            return self._client_cvs_dict[client_id]

    def _get_client_queue(self, client_id):
        with self._lock:
            return self._client_queues_dict[client_id]

    def put_request(self, client_id, request):
        # Here, get the mapping of the client to the GPU.
        # and place request in the workers request queue
        gpu, model, batch_size = self._control_manager.client_mapping(
            client_id)

        metadata = request[0]
        if gpu is not None:
            self._control_manager.update_request_metadata(client_id, metadata)
            self._worker_manager.schedule(gpu, request)
        else:
            # Drop the frame
            response = None
            self._get_client_queue(client_id).put((metadata, response))

    def get_response(self, client_id, timeout=-1):
        try:
            response = self._get_client_queue(client_id).get(timeout=timeout)
            next_model = self._control_manager.clients_next_model(client_id)
            response[0].next_model = next_model if next_model is not None else -1
        except queue.Empty:
            response = None
        # Get the current mapping from the control manager.
        # metadata.next_model = self._control_manager.client_mapping(client_id)
        # Do it here, or do in the _response_handler.
        # Multiple thread vs single thread access overhead
        return response

    def _response_handler(self):
        logging.info("Data dispatcher response handler started!")
        while not self._close_event.is_set():
            num_gpus = self._worker_manager.num_gpus
            # Iterate through workers in a round robin manner.
            for gpu in range(num_gpus):
                response = self._worker_manager.get_response(
                    gpu, non_block=True)
                if response is not None:
                    # print("Received response from worker")
                    metadata = response[0]
                    client_id = metadata.client_id
                    self._get_client_queue(client_id).put(response)
            time.sleep(0.001)

    def collect_request_stats(self, client_id, request_size, bw, metadata):
        self._control_manager.save_request_stats(
            client_id, request_size, bw, metadata)

    def collect_response_stats(self, client_id, response_size, metadata):
        # May be we need to collect stats such as transfer time to workers,
        # preprocessing time, waiting time.
        self._control_manager.save_response_stats(
            client_id, response_size, metadata)

        # This function is called by response thread of every client.
        # So, multiple threads would processing their stats.
        self._handle_stats(metadata)

    def _handle_stats(self, metadata):
        send_net_time = (metadata.server_recv_ts -
                         metadata.client_send_ts) * 1e3
        dispatch_time = (metadata.worker_recv_ts -
                         metadata.server_recv_ts) * 1e3
        processing_time = (metadata.worker_processing_ts -
                           metadata.worker_recv_ts) * 1e3
        waiting_time = (metadata.worker_picked_ts -
                        metadata.worker_processing_ts) * 1e3
        prediction_time = (metadata.worker_done_ts -
                           metadata.worker_picked_ts) * 1e3

        # XXX: Remember that, the logs are not in the sorted order of frame ids.
        stats = {
            "client_id": metadata.client_id,
            "frame_id": metadata.frame_id,
            "send_net_time": round(send_net_time, 3),
            "dispatch_time": round(dispatch_time, 3),
            "processing_time": round(processing_time, 3),
            "waiting_time": round(waiting_time, 3),
            "prediction_time": round(prediction_time, 3),
            'client_bw': round(metadata.client_bw, 3),
            'estimated_bw': round(metadata.estimated_bw, 3),
            'request_size': metadata.request_size,
            'response_size': metadata.response_size,
            'scheduler_run_count': metadata.scheduler_run_count,
            'used_model': metadata.used_model,
            'gpu_number': metadata.gpu_number
        }
        self._stats_logger.log(stats)
