import queue
import sys
import time
from typing import Iterable
from src.server.request_dispatcher import RequestDispatcher
import grpc
from src.protos import predict_pb2
from src.protos import predict_pb2_grpc
from concurrent import futures
import threading
import pickle
import logging

_MAX_MSG_LENGTH = 100 * 1024 * 1024
START_FRAME, END_FRAME, ID = 0, 1, 2

metadata_fields = ['client_id', 'frame_id', 'client_send_ts',
                   'server_recv_ts', 'worker_recv_ts', 'worker_processing_ts',
                   'worker_picked_ts', 'worker_done_ts', 'desired_model',
                   'used_model', 'next_model', 'gpu_number',
                   'request_size', 'response_size', 'client_bw',
                   'estimated_bw', 'scheduler_run_count', 'time_on_network', 'slo',
                   'model_predict_time', 'model_batch_size']


class FrameMetadata(object):
    __slots__ = metadata_fields

    def __init__(self) -> None:
        self.client_id = 0
        self.frame_id = 0
        self.client_send_ts = 0.0
        self.server_recv_ts = 0.0
        self.worker_recv_ts = 0.0
        self.worker_processing_ts = 0.0
        self.worker_picked_ts = 0.0
        self.worker_done_ts = 0.0
        self.desired_model = 0
        self.used_model = -1
        self.next_model = -1
        self.gpu_number = -1

        # TODO: Large number of fields might slow down the queue
        # transfer between two processes.
        self.request_size = 0
        self.response_size = 0
        self.client_bw = 0.0
        self.estimated_bw = 0.0
        self.scheduler_run_count = 0
        self.time_on_network = 0.0
        self.slo = 0
        self.model_predict_time = 0.0
        self.model_batch_size = 0


class ClientStub(object):
    def __init__(self, client_id) -> None:
        self.client_id = client_id
        self.stream_active_event = threading.Event()
        self.ack_queue = queue.Queue()


class ModelServingServicer(predict_pb2_grpc.ModelServingServicer):
    """
    gRPC server for ModelServing Service
    """

    def __init__(self, dispatcher: RequestDispatcher, stats=None):
        self._dispatcher = dispatcher
        self._lock = threading.Lock()

        # Maintain clients stub to handle clients specific resources
        self._clients_dict = {}

    def register(self, request, context):
        client_info = request.client_info
        client_id, model_number = self._dispatcher.register_client(slo=client_info.slo,
                                                                   frame_rate=client_info.frame_rate,
                                                                   lat_wire=client_info.lat_wire,
                                                                   init_bw=client_info.init_bw)
        client_stub = ClientStub(client_id)
        with self._lock:
            self._clients_dict[client_id] = client_stub

        client_token = predict_pb2.ClientToken(client_id=client_id)
        return predict_pb2.RegisterResponse(client_token=client_token, model_number=model_number)

    def unregister(self, request, context):
        self._dispatcher.unregister_client(request.client_id)
        with self._lock:
            del self._clients_dict[request.client_id]

        return predict_pb2.Empty()

    def predict(self, request_iter: Iterable[predict_pb2.PredictRequest],
                context: grpc.ServicerContext) -> Iterable[predict_pb2.PredictResponse]:
        """
        Implementation of the rpc predict service declared in the proto
        file above.
        """
        def _handle_requests():
            logging.info("Request handler started")
            n_requests = 0
            client_id, client_stub = None, None
            for request in request_iter:
                if client_id is None:
                    client_id = request.client_token.client_id
                    with self._lock:
                        client_stub = self._clients_dict[client_id]
                        client_stub.stream_active_event.set()

                    with shared_data_lock_cv:
                        shared_data["client_id"] = client_id
                        shared_data_lock_cv.notify()

                frame_id = request.frame_meta.frame_id
                client_send_ts = request.frame_meta.send_timestamp
                server_recv_ts = time.time()

                metadata = FrameMetadata()
                metadata.client_id = client_id
                metadata.frame_id = frame_id
                metadata.client_send_ts = client_send_ts
                metadata.server_recv_ts = server_recv_ts
                metadata.desired_model = request.frame_meta.desired_model
                metadata.client_bw = request.frame_meta.bw
                metadata.request_size = request.ByteSize()

                frame_data = request.frame_data.img
                n_requests = n_requests + 1

                logging.debug(f"Received request frame {metadata.frame_id}"
                              f" for client_{metadata.client_id}")
                client_stub.ack_queue.put((frame_id, server_recv_ts))
                self._dispatcher.put_request(client_id, (metadata, frame_data))
                self._dispatcher.collect_request_stats(client_id, request.ByteSize(),
                                                       request.frame_meta.bw,
                                                       metadata)
                logging.debug(f"Finished request frame {metadata.frame_id}"
                              f" for client_{metadata.client_id}")
            with shared_data_lock_cv:
                shared_data["total_requests"] = n_requests
            client_stub.stream_active_event.clear()
            client_stub.ack_queue.put(None)
            logging.info(f"Request handler for client {client_id} stopped!")

        def _handle_response():
            logging.info("Response handler started")
            try:
                n_responses = 0
                total_responses = sys.maxsize
                with shared_data_lock_cv:
                    if shared_data["client_id"] == None:
                        shared_data_lock_cv.wait()
                    client_id = shared_data["client_id"]
                    client_stub = self._clients_dict[client_id]

                while n_responses < total_responses:
                    if not client_stub.stream_active_event.is_set():
                        total_responses = shared_data["total_requests"]
                    res = self._dispatcher.get_response(client_id, timeout=1.0)
                    if res == None:
                        # Timed out. Continue until we send all responses.
                        continue

                    # Received response for the frame. Generate grpc response and send.
                    metadata, frame_response = res
                    assert metadata.client_id == client_id
                    frame_meta = predict_pb2.FrameMeta(
                        frame_id=metadata.frame_id)
                    server_send_ts = time.time()
                    det_meta = predict_pb2.DetectionMeta(used_model=metadata.used_model,
                                                         send_timestamp=server_send_ts,
                                                         next_model=metadata.next_model)
                    client_token = predict_pb2.ClientToken(client_id=client_id)
                    response = predict_pb2.PredictResponse(frame_meta=frame_meta, client_token=client_token,
                                                           det_meta=det_meta,
                                                           data=pickle.dumps(frame_response))
                    self._dispatcher.collect_response_stats(
                        client_id, response.ByteSize(), metadata)
                    n_responses += 1
                    logging.debug(f"Sending response frame {metadata.frame_id}"
                                  f" for client_{metadata.client_id}")
                    yield response
            except Exception as e:
                logging.error("Response handler exception {}".format(e))
            logging.info(f"Response handler for client {client_id} stopped!")
            yield None

        logging.info(
            f"Bidirectional stream started for client {context.peer()}")
        # client_id = context.peer()
        shared_data = {"client_id": None, "total_requests": None}
        shared_data_lock_cv = threading.Condition()

        total_requests = 0
        threading.Thread(target=_handle_requests).start()

        response_handler = _handle_response()
        while True:
            response = next(response_handler)
            if response is None:
                break
            yield response
        logging.info(
            f"Bidirectional stream stopped for client {context.peer()}")

    def predict_ack(self, request, context):
        '''
        This uni-directional stream is created to ack back predict requests.
        This is done to mostly estimate bw on the client side.
        TODO: Revisit this design as it creates one more thread to serve the ack stream.
        The more correct approach is to retreive the size and RTT stats from the grpc on the client.
        '''
        client_id = request.client_id
        with self._lock:
            client_stub = self._clients_dict[client_id]
        while True:
            request_metadata = client_stub.ack_queue.get()
            if request_metadata is None:
                break
            frame_id, server_recv_req_ts = request_metadata
            request_ack = predict_pb2.PredictRequestAck(client_token=predict_pb2.ClientToken(client_id=client_id),
                                                        frame_id=frame_id,
                                                        server_recv_req_ts=server_recv_req_ts,
                                                        server_send_ack_ts=time.time())
            yield request_ack
        logging.info(f"Ack streaming stopped for client {client_id}")

    def reset(self, request, context):
        logging.info("Reset..")
        self.cache.reset()
        return predict_pb2.Empty()


class ModelServer(object):
    def __init__(self, dispatcher: RequestDispatcher, server_port=9999,
                 grpc_max_workers=10) -> None:
        """
        Function which actually starts the gRPC servers, and preps
        it for serving incoming connections
        """
        # declare a server object with desired number of thread pool workers.
        options = [('grpc.max_send_message_length', _MAX_MSG_LENGTH),
                   ('grpc.max_receive_message_length', _MAX_MSG_LENGTH),
                   ("grpc.keepalive_time_ms", 10000),
                   ("grpc.keepalive_timeout_ms", 5000),
                   ("grpc.keepalive_permit_without_calls", True),
                   ("grpc.http2.max_ping_strikes", 0),
                   ('grpc.http2.min_time_between_pings_ms', 10000),
                   ('grpc.http2.min_ping_interval_without_data_ms',  5000)]
        # ('grpc.max_connection_idle_ms', 5000)]
        self.model_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_max_workers), options=options)
        # compression=grpc.Compression.Gzip)

        # This line can be ignored
        predict_pb2_grpc.add_ModelServingServicer_to_server(
            ModelServingServicer(dispatcher), self. model_server)

        # bind the server to the port defined above
        self.model_server.add_insecure_port('[::]:{}'.format(server_port))

    def start_and_wait(self):
        # start the server
        self.model_server.start()
        logging.info('Model Serving Server running ...')
        self.model_server.wait_for_termination()

    def stop(self):
        self.model_server.stop(grace=1)
