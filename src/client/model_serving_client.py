import logging
import grpc
import src.protos.predict_pb2 as predict_pb2
import src.protos.predict_pb2_grpc as predict_pb2_grpc
import numpy as np
import cv2
import time
import threading
import queue
from src.utils import FRAME_SIZES

_MAX_MSG_LENGTH = 100 * 1024 * 1024
START_FRAME, END_FRAME, ID = 0, 1, 2


class RequestIterator(object):
    def __init__(self, req_queue: queue.Queue, close_event):
        self._req_queue = req_queue
        self._close_event = close_event

    def _next(self):

        if self._close_event.is_set():
            logging.info("Client Request Stopped!")
            raise StopIteration()

        request = self._req_queue.get()
        if request is None:
            logging.info("Client Request Stopped!")
            raise StopIteration()

        return request

    def __iter__(self):
        return self

    def __next__(self):
        return self._next()

    def next(self):
        return self._next()


class ModelServingClient(object):
    """
    Client for connecting to ModelServingServer
    """

    def __init__(self, host="localhost", port_number=9999, callback=None, stats=False, stats_name="model_client"):
        # configure the host and the the port to which the client should connect to.
        self.host = host
        self.server_port = port_number

        # instantiate a communication channel
        options = [('grpc.max_send_message_length', _MAX_MSG_LENGTH),
                   ('grpc.max_receive_message_length', _MAX_MSG_LENGTH),
                   ("grpc.keepalive_time_ms", 10000),
                   ("grpc.keepalive_timeout_ms", 5000),
                   ("grpc.keepalive_permit_without_calls", True),
                   ("grpc.http2.max_ping_strikes", 0),
                   ('grpc.http2.min_time_between_pings_ms', 10000),
                   ('grpc.http2.min_ping_interval_without_data_ms',  5000)]
        # ('grpc.max_connection_idle_ms', 5000)]
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port), options=options)
        # compression=grpc.Compression.Gzip)

        # bind the client to the server channel
        self.stub = predict_pb2_grpc.ModelServingStub(self.channel)

        # stats
        self.stats = None
        if stats:
            self.stats = utils.Stats(["preprocessing_time", "prediction_time",
                                      "postprocessing_time"], stats_name=stats_name)

        self._callback = callback

    def _init_bidi_stream(self):
        self.throttle_limit = 32  # outstanding requests
        self._request_queue = queue.Queue()
        self._req_lock_cv = threading.Condition()
        self._close_event = threading.Event()
        self._res_lock_cv = threading.Condition()

        # We can provide the request iterator as a class or the function.
        # self.responses = self.stub.predict(self._request_iterator())
        self.responses = self.stub.predict(RequestIterator(
            self._request_queue, self._close_event))

        self._reponse_thread = threading.Thread(
            target=self._handle_predict_response)
        self._reponse_thread.start()

        # Start receiving ack stream. TODO: Rehaul the protocol.
        # Send request as unary request. And have response as a stream
        self._request_acks = self.stub.predict_ack(self._client_token_pb)
        self._request_ack_thread = threading.Thread(
            target=self._handle_request_ack)
        self._request_ack_thread.start()

    def _stop_bidi_stream(self):
        self._close_event.set()
        # Stop request iterator
        self._request_queue.put(None)
        # Stop response thread
        self._reponse_thread.join()
        # Stop request ack thread
        self._request_ack_thread.join()

    def register_with_server(self, slo, frame_rate, lat_wire, init_bw):
        client_info = predict_pb2.ClientInfo(slo=slo, frame_rate=frame_rate,
                                             lat_wire=lat_wire, init_bw=init_bw)
        register_response_pb = self.stub.register(
            predict_pb2.RegisterRequest(client_info=client_info))
        self._client_token_pb = register_response_pb.client_token
        logging.info(
            f"Registered client {self._client_token_pb.client_id} with the server!")
        self._init_bidi_stream()
        return self._client_token_pb.client_id, register_response_pb.model_number

    def unregister_with_server(self):
        self._stop_bidi_stream()
        self.stub.unregister(self._client_token_pb)
        self.channel.close()

    def register_callback(self, callback):
        self._callback = callback

    @staticmethod
    def resize_and_encode(img, frame_size):
        _ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        img = cv2.resize(img, (frame_size, frame_size), cv2.INTER_NEAREST)
        img_bytes = cv2.imencode(".jpg", img, _ENCODE_PARAM)[1].tobytes()
        return img_bytes

    def serialize(self, desired_model, frame_data):
        bytes = self.resize_and_encode(
            frame_data, FRAME_SIZES[desired_model])
        return bytes

    def unserialize(self, output, output_size):
        output = np.frombuffer(output, dtype=np.float32).reshape(output_size)
        return output

    def _handle_predict_response(self):
        '''
        Handle response from model server.
        '''

        def _process_response(model_output):
            frame_meta = model_output.frame_meta
            client_token = model_output.client_token

            # TODO: replace this dictionary with tuple
            assert client_token.client_id == self._client_token_pb.client_id
            output = {
                "frame_id": frame_meta.frame_id,
                "used_model": model_output.det_meta.used_model,
                "next_model": model_output.det_meta.next_model,
                "server_send_res_ts": model_output.det_meta.send_timestamp,
                "client_recv_res_ts": time.time(),
                "data": model_output.data
            }
            return output

        logging.info("Started response receiver!")
        while True:
            if self._close_event.is_set():
                break
            try:
                for response in self.responses:
                    output = _process_response(response)
                    frame_id = output["frame_id"]

                    # Call upper layer callback. If this is the next expected result.
                    # Use frame_id sequence to know if this is the next expected result.
                    try:
                        self._res_lock_cv.acquire()
                        self._callback(frame_id, output)
                        self._res_lock_cv.notify()
                    finally:
                        self._res_lock_cv.release()
            except Exception as e:
                logging.error(f"Execption in response: {str(e)}")

        return

    def _handle_request_ack(self):
        '''
        Handler to request ack stream.
        '''

        def _process_acks(ack):
            client_token = ack.client_token
            assert client_token.client_id == self._client_token_pb.client_id
            # TODO: replace this dictionary with tuple
            output = {
                "frame_id": ack.frame_id,
                "server_recv_req_ts": ack.server_recv_req_ts,
                "server_send_ack_ts": ack.server_send_ack_ts,
                "client_recv_ack_ts": time.time()
            }
            return output

        logging.info("Started response receiver!")
        while True:
            if self._close_event.is_set():
                break
            try:
                for acks in self._request_acks:
                    output = _process_acks(acks)
                    frame_id = output["frame_id"]

                    # Call upper layer callback. If this is the next expected result.
                    # Use frame_id sequence to know if this is the next expected result.
                    try:
                        self._callback.handle_acks(frame_id, output)
                    except Exception as e:
                        logging.error(f"Exception in handle acks {str(e)}")
            except Exception as e:
                logging.error(f"Exception in handle acks: {str(e)}")
                exit()

        return

    def _throttle(self):
        if self._callback is None:
            return
        try:
            self._res_lock_cv.acquire()
            while self._callback.outstanding_count() > self.throttle_limit:
                self._res_lock_cv.wait()
            self._res_lock_cv.notify()
        finally:
            self._res_lock_cv.release()

        return

    def _request_iterator(self):
        try:
            while not self._close_event.is_set():
                request = self._request_queue.get()
                if request is None:
                    break
                yield request
        except Exception as e:
            logging.error(f"Exception in request iterator: {str(e)}")
        logging.info("Client Request Stopped!")
        raise StopIteration()

    def predict(self, metadata, frame_data):
        """
        Client function to call the rpc for predict
        """
        # Serialize input
        input_data = self.serialize(metadata.desired_model, frame_data)

        # send_timestamp = time.time()
        frame_meta = predict_pb2.FrameMeta(frame_id=metadata.frame_id,
                                           desired_model=metadata.desired_model,
                                           # send_timestamp=send_timestamp,
                                           bw=metadata.bw)

        # client_info = predict_pb2.ClientInfo(client_id=0)
        frame_data = predict_pb2.FrameData(img=input_data)
        predict_request = predict_pb2.PredictRequest(client_token=self._client_token_pb,
                                                     frame_meta=frame_meta,
                                                     frame_data=frame_data)

        # Check throttle
        self._throttle()
        metadata.frame_wire_size = predict_request.ByteSize()
        metadata.client_send_req_ts = time.time()
        predict_request.frame_meta.send_timestamp = metadata.client_send_req_ts
        self._request_queue.put(predict_request)

        return None

    def reset(self):
        empty = predict_pb2.Empty()
        response = self.stub.reset(empty)

    def __call__(self, frame):
        return self.predict(frame)
