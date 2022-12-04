import logging
import threading
import pickle
import time
import json
import src.utils as utils
import os


class ResponseHandler(threading.Thread):
    def __init__(self, opts, controller_client, fps_logger, stats_logger, log_path) -> None:
        self.opts = opts
        self._controller_client = controller_client
        self._close_event = threading.Event()
        self._handler_thread = threading.Thread(target=self._run)
        self._lock = threading.Lock()
        self._fps_logger = fps_logger
        self._stats_logger = stats_logger
        self._log_path = log_path
        self.reset()

        self._handler_thread.start()

    def reset(self):
        self._outstanding_req = {}
        self._responses = {}
        self._response_id = 0
        self._output_dets_json = []

    def close(self):
        self._close_event.set()
        self._handler_thread.join()
        self.save_output_dets()

    def outstanding_count(self):
        with self._lock:
            count = len(self._outstanding_req)
        return count

    def add_outstanding_req(self, metadata):
        with self._lock:
            self._outstanding_req[metadata.frame_id] = metadata

    def save_output_dets(self):
        sorted_dets = list(sorted(
            self._output_dets_json, key=lambda x: x["image_id"]))
        sorted_dets = list(
            map(utils.convert_cat_id_and_reorientate_bbox, sorted_dets))
        print(sorted_dets)
        with open(os.path.join(self._log_path, "output_dets.json"), 'w',
                  encoding='utf-8') as file:
            json.dump(sorted_dets, file, indent=2)

    def _handle_response(self, response):
        def handle_stats(dropped_frame, metadata):
            send_timestamp = utils.timestamp_to_str(
                metadata.client_send_req_ts)
            recv_timestamp = utils.timestamp_to_str(
                metadata.client_recv_res_ts)
            e2e_latency = round(
                (metadata.client_recv_res_ts - metadata.client_send_req_ts) * 1e3, 3)
            self._stats_logger.log({'client_id': metadata.client_id,
                                    'frame_id': metadata.frame_id,
                                    'send_ts': send_timestamp,
                                    'recv_ts': recv_timestamp,
                                    'e2e_latency': e2e_latency,
                                    'network_time': round(metadata.network_time, 3),
                                    'frame_wire_size': metadata.frame_wire_size,
                                    'desired_model': metadata.desired_model,
                                    'used_model': metadata.used_model,
                                    'next_model': metadata.next_model,
                                    'dropped_frame': dropped_frame})
            missed_deadline_flag = (e2e_latency > self.opts.slo)
            return missed_deadline_flag

        def handle_output_det(det_boxes, metadata):
            dets = []
            for det_box in det_boxes:
                x1 = round(det_box[0] * metadata.orig_image_width, 2)
                y1 = round(det_box[1] * metadata.orig_image_height, 2)
                x2 = round(det_box[2] * metadata.orig_image_width, 2)
                y2 = round(det_box[3] * metadata.orig_image_height, 2)
                w = round(x2 - x1, 2)
                h = round(y2 - y1, 2)
                bbox = [x1, y1, w, h]  # COCO format.
                det = {'image_id': metadata.frame_id,
                       'category_id': int(det_box[-1]),
                       'bbox': bbox,
                       'score': round(float(det_box[-2]), 2),
                       'image_height': metadata.orig_image_height,
                       'image_width': metadata.orig_image_width
                       }
                dets.append(det)
            with self._lock:
                self._output_dets_json.extend(dets)
            return

        if response is None:
            return

        metadata, output = response
        assert metadata.frame_id == output["frame_id"]
        metadata.client_recv_res_ts = output["client_recv_res_ts"]
        metadata.server_send_res_ts = output["server_send_res_ts"]
        metadata.next_model = output['next_model']
        metadata.used_model = output['used_model']

        # Handle actual boxes output
        det_boxes = pickle.loads(output["data"])
        dropped_frame = 0
        if det_boxes is None:
            dropped_frame = 1

        #  Update next model
        self._controller_client.update_next_model(metadata.next_model)

        # Collect stats
        missed_deadline_flag = handle_stats(dropped_frame, metadata)

        # handler output detections from the server
        if dropped_frame or missed_deadline_flag:
            # The prediction request was dropped on the server.
            logging.debug(f"Dropped ({dropped_frame}) or deadline missed ({missed_deadline_flag})"
                        f" frame {output['frame_id']}")
        else:
            logging.debug(f"Output boxes shape {det_boxes.shape} frame_id {output['frame_id']},"
                          f" used model {output['used_model']}")
            handle_output_det(det_boxes, metadata)
        return

    def _run(self):
        logging.info("Response handler thread has started!")
        try:
            while not self._close_event.is_set():
                response = None
                with self._lock:
                    if self._response_id in self._responses:
                        response = self._responses[self._response_id]
                        del self._responses[self._response_id]
                        print(
                            f"Handling response for frame {self._response_id}")
                        self._response_id += 1

                # TODO: Instead of sleep, have a wait and notify
                time.sleep(0.001)
                self._handle_response(response)
        except Exception as e:
            logging.error(f"Execption in response handler: {str(e)}")

    def __call__(self, response_id, response):
        if response_id not in self._outstanding_req:
            raise RuntimeError("Wrong response received")

        print(f"Received response from server for frame {response_id}")
        self._fps_logger.update()
        # May be replace this with a queue
        with self._lock:
            metadata = self._outstanding_req[response_id]
            self._responses[response_id] = (metadata, response)
            del self._outstanding_req[response_id]

    def handle_acks(self, response_id, response):
        metadata = None
        with self._lock:
            if response_id in self._outstanding_req:
                metadata = self._outstanding_req[response_id]

        if metadata is None:
            return

        metadata.server_recv_req_ts = response["server_recv_req_ts"]
        metadata.server_send_ack_ts = response["server_send_ack_ts"]
        metadata.client_recv_ack_ts = response["client_recv_ack_ts"]

        self._controller_client.update_bw(metadata)
