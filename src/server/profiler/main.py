import logging
from src.server.model_server import FrameMetadata
import threading
import numpy as np
from src.server.request_dispatcher import RequestDispatcher
from src.server.worker.manager import WorkerManager
from src.server.profiler.opts import parse_args
import signal
import sys
from src.utils import FRAME_SIZES, setup_logging, sleepInMillis
import time
import random
import os
import json
import cv2
import gc
gc.disable()


def close_processes():
    request_dispatcher.close()
    control_manager.close()
    worker_manager.close()


def signal_handler(signum, frame):
    close_processes()
    sys.exit(0)


def initial_profiled_latencies(n_models, max_batch_size):
    # These profiled latecies won't be used by the worker.
    # Therefore, return zero matrix.
    return np.zeros((n_models, max_batch_size))


def initial_profiled_accuracy(n_models):
    return np.zeros(n_models)


class ControlManager(object):
    '''
    Simple controller which maps clients to the gpus.
    '''

    def __init__(self, opts, worker_manager):
        self.opts = opts
        self._worker_manager = worker_manager
        self._lock = threading.Lock()
        self._client_id_idx = 0

    def client_mapping(self, client_seq):
        # We use clients = GPUs. And map client_id to the gpu id.
        return client_seq, client_seq, None

    def add_client(self, slo, frame_rate, lat_wire, init_bw):
        with self._lock:
            client_id = self._client_id_idx
            self._client_id_idx += 1
        return client_id, 0

    def remove_client(self, client_id):
        self._client_id_idx -= 1

    def update_request_metadata(self, client_seq, metadata):
        metadata.time_on_network = 1.0

    def clients_next_model(self, client_seq):
        return 0

    def change_worker_model(self, model_number, batch_size):
        # Request worker to change the model number and the batch size
        [self._worker_manager.update_worker_model(gpu_id, model_number, batch_size)
         for gpu_id in range(self.opts.n_gpus)]

    def close(self):
        pass


class LoadGenerator(threading.Thread):
    '''
    LoadGenerator per GPU to feed Worker process with enough number of requests.
    '''

    def __init__(self, model_number, batch_size, dataset_dir,
                 total_batch_iter, gt_annotations_path, profile_dir,
                 request_dispatcher: RequestDispatcher):
        threading.Thread.__init__(self)

        self._model_number = model_number
        self._batch_size = batch_size
        self._dataset_dir = dataset_dir
        self._total_img_iter = total_batch_iter * self._batch_size
        self._dispatcher = request_dispatcher
        self.images = self.read_images_annotations(gt_annotations_path)
        self._profile_dir = profile_dir
        self._total_images = len(self.images)
        self._throttle_limit = self._batch_size * 2
        self._outstanding_req_count = 0
        self._lock = threading.Lock()

        self._client_id, _ = self._dispatcher.register_client(
            slo=15000, frame_rate=25, lat_wire=1.0, init_bw=10000)
        self._response_handler = threading.Thread(
            target=self._run_response_handler)
        self._response_handler.start()

    @staticmethod
    def read_images_annotations(gt_annotations_path):
        annotations_file_path = gt_annotations_path
        with open(annotations_file_path) as annotations_file:
            try:
                annotations = json.load(annotations_file)
            except:
                print("annotations file not a json")
                exit()
        return annotations['images']

    @staticmethod
    def read_image_in_jpg(dataset_dir, frame_size, total_images, images):
        _ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        # index = index % total_images
        index = random.randint(0, total_images-1)
        image_file_name = images[index]["file_name"]

        img = cv2.imread(os.path.join(dataset_dir, image_file_name))
        img = cv2.resize(img, (frame_size, frame_size), cv2.INTER_NEAREST)
        jpg_file = cv2.imencode(".jpg", img, _ENCODE_PARAM)[1].tobytes()

        return jpg_file

    def _get_metadata(self, frame_id, request):
        metadata = FrameMetadata()
        metadata.client_id = self._client_id
        metadata.frame_id = frame_id
        metadata.client_send_ts = time.time()
        metadata.server_recv_ts = time.time()
        metadata.desired_model = self._model_number
        metadata.client_bw = 100 * 1024
        metadata.request_size = 10000
        metadata.slo = 15000
        return metadata

    def _wait_for_outstanding(self):
        while True:
            with self._lock:
                if self._outstanding_req_count < self._throttle_limit:
                    self._outstanding_req_count += 1
                    return
            sleepInMillis(1)

    def run(self):
        logging.info(f"Started load generator {self._client_id} "
                     f"with model {self._model_number} and batch size {self._batch_size}")
        for frame_id in range(self._total_img_iter):
            # Read next frame request
            request = self.read_image_in_jpg(self._dataset_dir,
                                             FRAME_SIZES[self._model_number],
                                             self._total_images, self.images)

            # generate metadata
            metadata = self._get_metadata(frame_id, request)

            # Wait for outstanding requests. Don't overload worker.
            self._wait_for_outstanding()

            # Dispatch the request
            self._dispatcher.put_request(
                metadata.client_id, (metadata, request))

        # Wait for all responses to be received
        self._response_handler.join()

        self._dispatcher.unregister_client(self._client_id)
        logging.info(f"Stopped load generator {self._client_id} with model"
                     f" {self._model_number} and batch size {self._batch_size}")

    def _run_response_handler(self):
        logging.info(f"Started response handler of load generator {self._client_id}"
                     f" with model {self._model_number} and batch size {self._batch_size}")
        frame_size = FRAME_SIZES[self._model_number]
        predict_time_lst = []
        batch_size_lst = []
        prev_predict_time = 0
        for _ in range(self._total_img_iter):
            metadata, _ = self._dispatcher.get_response(
                self._client_id, timeout=None)
            assert metadata.client_id == self._client_id
            assert metadata.gpu_number == self._client_id
            with self._lock:
                self._outstanding_req_count -= 1

            # Avoid duplicate predict times.
            # We have to record batch timings, and multiple frames can be part of the batch.
            # Therefore, we need to avoid duplicate predict time, piggybacked on every frame response.
            predict_time = metadata.model_predict_time
            batch_size = metadata.model_batch_size
            if prev_predict_time != predict_time:
                predict_time_lst.append(predict_time)
                batch_size_lst.append(batch_size)
            prev_predict_time = predict_time

        #  Dump the predictions to a file
        profile_dir = os.path.join(
            self._profile_dir, f'profiles_gpu_{self._client_id}')
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
        output_file = open(
            f'{profile_dir}/profile_latency_{frame_size}.txt'.format(frame_size), 'a')

        if self._batch_size == 1:  # Add header line
            print("{:<20s},{:<20s},{:<20s}".format(
                "ModelSize", "Batch", "InferenceTime"), file=output_file)

        print(f"Saving prediction time of gpu {self._client_id}"
              f" for iterations: {len(predict_time_lst)}")
        warmup_count = 10
        for i in range(warmup_count, len(predict_time_lst)):
            predict_time = predict_time_lst[i]
            batch_size = batch_size_lst[i]
            print("{:<20d},{:<20d},{:<20.2f}".format(
                frame_size, batch_size, predict_time), file=output_file)
        output_file.close()


if __name__ == '__main__':
    # Parse command line arguments
    opts = parse_args()

    # Setup logging
    setup_logging(opts, log_name="main_profiler")

    # Read profiled values
    profiled_latencies = initial_profiled_latencies(
        opts.n_models, opts.max_batch_size)
    profiled_accuracies = initial_profiled_accuracy(opts.n_models)

    # Start control manager, workers and dispatcher
    worker_manager = WorkerManager(opts, profiled_latencies)
    control_manager = ControlManager(opts, worker_manager)

    request_dispatcher = RequestDispatcher(
        opts, worker_manager, control_manager)

    # Setup signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait to workers to load models
    time.sleep(10)

    # Start load generators per GPU
    logging.info("Load generators started!")
    for batch_size in range(1, opts.max_batch_size+1):
        logging.info(
            f"Profiling model {opts.init_model_number} with batch size {batch_size}")

        #  Change batch size of the worker
        control_manager.change_worker_model(opts.init_model_number, batch_size)

        # wait for workers to adjust to new batch size
        time.sleep(1)

        # Start load generators
        load_generators = []
        for gpu_id in range(opts.n_gpus):
            load_generator = LoadGenerator(opts.init_model_number, batch_size,
                                           opts.dataset_dir, opts.total_profile_iter, opts.gt_annotations_path,
                                           opts.profile_dir,
                                           request_dispatcher)
            load_generator.start()
            load_generators.append(load_generator)

        # Wait for load generators to finish
        for load_generator in load_generators:
            load_generator.join()

    close_processes()
    logging.info("Load generators stopped!")
