import logging
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
import queue
import time
import torch
import src.server.worker.model_utils as model_utils
import threading
import numpy as np
import src.utils as utils
import numpy as np
import src.server.queue_utils as queue_utils

_EXECUTOR_TYPE = "process"  # ["thread", "process"]


def _executor(simulate_gpu, gpu_number, profiled_latencies,
              processed_frames_queue, output_queue, active_model_queue,
              close_event):
    ''' The executor function to execute requests with active models '''

    def _waiting_time_drop(metadata):
        '''Drop requests when the waiting time exceeds the model execution time.

            There are two options. 1. Include the preprocessing time in the waiting time
            2. or Don't include. TODO: Handle preprocessing time.
        '''
        waiting_time = (metadata.worker_picked_ts -
                        metadata.worker_processing_ts) * 1e3
        if (waiting_time - model_time) > 1.0:
            return True
        return False

    def _lazy_drop(metadata):
        dispatch_time = (metadata.worker_recv_ts -
                         metadata.server_recv_ts) * 1e3
        processing_time = (metadata.worker_processing_ts -
                           metadata.worker_recv_ts) * 1e3
        waiting_time = (metadata.worker_picked_ts -
                        metadata.worker_processing_ts) * 1e3
        elapsed_time = metadata.time_on_network + 2 * \
            dispatch_time + processing_time + waiting_time
        remaining_time = metadata.slo - elapsed_time

        print(f"LAZY_DROP {metadata.frame_id}: {metadata.time_on_network}, {dispatch_time}, {processing_time},"
              f" {waiting_time}, {elapsed_time}, remaining_time:{remaining_time}, model_time:{model_time}")
        if (model_time - remaining_time) > 3.0:
            return True, None
        return False, (time.time(), remaining_time)

    def drop_requests(metadata):
        _DROP_STRATEGY = "LAZY_DROP"  # ["WAITING_TIME", "LAZY_DROP"]
        if _DROP_STRATEGY == "WAITING_TIME":
            drop_flag = _waiting_time_drop(metadata)
        elif _DROP_STRATEGY == "LAZY_DROP":
            drop_flag, remaining_time = _lazy_drop(metadata)

        if drop_flag:
            print(f"Dropped frame {metadata.frame_id} for "
                  f"client {metadata.client_id} on gpu {gpu_number}")
            metadata.gpu_number = gpu_number
            output_queue.put((metadata, None))

        return drop_flag, remaining_time

    def get_batch_imgs(batch_size, model_number):
        def curr_remaining_time(remaining_time):
            curr_time = time.time()
            ret = remaining_time[1] - (curr_time - remaining_time[0]) * 1e3
            return ret

        def update_min_remaining_time(min_remaining_time, remaining_time):
            if min_remaining_time is None:
                return remaining_time

            r1 = curr_remaining_time(min_remaining_time)
            r2 = curr_remaining_time(remaining_time)
            if r2 < r1:
                return remaining_time
            else:
                return min_remaining_time

        metadata_lst, imgs = [], []
        idx = 0
        min_remaining_time = None
        while idx < batch_size:
            try:
                if processed_frames_queue.empty():
                    if min_remaining_time is not None and \
                            (curr_remaining_time(min_remaining_time) - model_time) > 10.0:
                        continue
                    break
                (metadata, img) = processed_frames_queue.get()
                metadata.worker_picked_ts = time.time()

                # Check if we should drop the request to avoid cascading effect
                # TODO: Need to be accurate here.
                # if drop_requests(metadata):
                drop_flag, remaining_time = drop_requests(metadata)
                if drop_flag:
                    continue

                min_remaining_time = update_min_remaining_time(
                    min_remaining_time, remaining_time)
                metadata_lst.append(metadata)
                img = model_utils.unserialize(img)
                img = model_utils.model_input_resize(model_number, img)
                imgs.append(img)
                idx += 1
            except queue.Empty:
                break

        if len(imgs) > 0:
            dummy_count = batch_size - len(imgs)
            for _ in range(dummy_count):
                imgs.append(imgs[0])
            np_imgs = np.array(imgs)
        else:
            '''
            XXX: This is a workaround for the following issue.

            We saw a great variation in the inference timings when GPU is not kept busy. That is,
            when we sleep for some time in between consecutive model inferences, the inference time greatly varies.
            This may be because of a *hidden* pytorch feature/bug (we don't know yet) or GC which tries to free some
            unreferenced objects/buffers and when we make an inference call again, it may allocate the objects/buffers again.
            But this is again a guess. There are some open issues reported but not addressed yet.

            ref: https://discuss.pytorch.org/t/different-latency-using-same-model-in-two-apps/99097
                 https://gitmemory.com/issue/pytorch/pytorch/44504/690822011
                 https://discuss.pytorch.org/t/inconsistent-model-inference-time/106933/9
            '''
            np_imgs = np.zeros(
                (batch_size, utils.FRAME_SIZES[model_number], utils.FRAME_SIZES[model_number], 3), dtype=np.uint8)
            dummy_count = batch_size

        return metadata_lst, np_imgs, dummy_count

    def get_active_model():
        '''Check if execution with the new active model is requested.'''
        model_number, batch_size, model = None, None, None
        try:
            if not active_model_queue.empty():
                (model_number, batch_size, model) = active_model_queue.get()
                active_model_queue.task_done()
        except queue.Empty:
            pass
        return model_number, batch_size, model

    def send_response(model_number, metadata_lst, batch_output, dummy_count, predict_time):
        # TODO: This can go in a seperate thread
        batch_size = len(batch_output)
        if dummy_count == batch_size:
            assert len(metadata_lst) == 0, "metadata_lst is not empty"
            return

        assert len(metadata_lst) == (batch_size - dummy_count)
        for i in range(0, batch_size - dummy_count):
            # each box is of size 7 float values
            boxes = np.array(batch_output[i])  # shape(num_boxes, 7)
            metadata = metadata_lst[i]
            metadata.worker_done_ts = time.time()
            metadata.used_model = model_number
            metadata.gpu_number = gpu_number
            metadata.model_predict_time = predict_time
            metadata.model_batch_size = batch_size

            output_queue.put((metadata, boxes))
        return

    def wait_for_ms(timeout):
        current_time = time.time()
        end_time = current_time + timeout * 1e-3
        while current_time <= end_time:
            # time.sleep(1)
            current_time = time.time()
            pass

    # Run model at periodic interval with batch size
    logging.info(f"Worker executor on gpu {gpu_number} started!")
    utils.set_process_nice(nice_value=-20)
    model_utils.set_deterministic_behaviour(seed=1)
    cuda_stream = torch.cuda.Stream(gpu_number)
    # cuda_stream = torch.cuda.default_stream(gpu_number)
    start_time = time.time()

    # Get the initial model
    model_number, batch_size, model = get_active_model()
    assert model_number is not None

    try:
        while not close_event.is_set():
            res = get_active_model()
            if res[0] is not None:
                model_number, batch_size, model = res

            # XXX: Always pay attention when indexing with batch_size.
            model_time = profiled_latencies[model_number][batch_size-1]

            _time = time.time()
            total_time = (_time - start_time) * 1e3
            start_time = _time
            metadata_lst, imgs, dummy_count = get_batch_imgs(batch_size,
                                                             model_number)
            batch_time = (time.time() - start_time) * 1e3
            if imgs.size != 0 and dummy_count != batch_size:
                with torch.cuda.stream(cuda_stream):
                    if not simulate_gpu:
                        batch_output = model_utils.model_predict(
                            model, imgs, use_gpu=1, gpu_number=gpu_number)
                    else:
                        batch_output = model.predict(
                            imgs, model_time, start_time)

                predict_time = (time.time() - start_time)*1e3
                # if not dummy_batch:
                send_response(model_number, metadata_lst, batch_output,
                              dummy_count, predict_time)

                print(f"Executor {gpu_number} (dummy={dummy_count}): predicting with model number"
                      f" {model_number} and batch size {batch_size} and img count {len(imgs)},"
                      f" total_predict_time {predict_time}, model_time {predict_time - batch_time},"
                      f" profiled_time {model_time}, batch time {batch_time},"
                      f" other_time {total_time}")

            else:
                '''
                if no requests have arrived then we may execute the dummy requests to keep GPU
                busy and to avoid the issue which is mentioned in the `get_batch_imgs()` function.
                But this will create a execution cycle type processing, like CPU cycles. This might have impact
                on the requests, if requests arrives in burst (close to each other).
                '''
                pass

    except (KeyboardInterrupt, SystemExit):
        pass

    logging.info(f"Worker executor on gpu {gpu_number} stopped!")


class Worker(mp.Process):
    def __init__(self, opts, gpu_number, profiled_latencies) -> None:
        super(Worker, self).__init__()
        self.opts = opts
        self._gpu_number = gpu_number
        self._profiled_latencies = profiled_latencies

        # Things declared here will be shared with the parent process.
        # self._input_queue = zmq_utils.Queue(
        #     name=f"input_queue", gpu_number=gpu_number)
        self._input_queue = mp.SimpleQueue()
        self._output_queue = mp.SimpleQueue()
        self._control_input_queue = mp.SimpleQueue()

        # alloc events to signal
        self._close_event = mp.Event()

    def put_request(self, request):
        ''' Entry point of all requests to this worker '''
        self._input_queue.put(request)

    def get_response_nowait(self):
        ''' Exit point of all responses from this worker '''
        response = None
        try:
            if not self._output_queue.empty():
                response = self._output_queue.get()
        except queue.Empty:
            pass
        return response

    def close(self):
        self._close_event.set()
        self._input_queue.put(None)
        self._control_input_queue.put(None)

    def _stop_process(self):
        ''' Stop all process local things here '''
        # Stop threads
        self._executor_thread.join()
        self._loader_thread.join()

        # Close local queues
        self._stop_local_queues()

    def _init_process(self):
        ''' Initialize all process local things here '''
        # Setup logging
        utils.setup_logging(
            self.opts, log_name="worker_{}".format(self._gpu_number))

        # 1. Load all models into cpu memory.
        self._models = self._load_all_models_cpu()
        self._lock = threading.Lock()

        # 2. Init local queues
        self._init_local_queues()

        # 3. load initial actve model on the GPU
        self._load_init_models_cuda()

        # 4. Start all threads
        self._start_threads()

    def _init_local_queues(self):
        if _EXECUTOR_TYPE == "thread":
            self._processed_frames_queue = queue.Queue()
            self._active_model_queue = queue.Queue()
        else:
            # self._processed_frames_queue = mp.SimpleQueue()
            self._processed_frames_queue = queue_utils.ParallelQueue(
                mp.SimpleQueue, count=4)
            self._active_model_queue = mp.JoinableQueue()

    def _stop_local_queues(self):
        if _EXECUTOR_TYPE == "thread":
            # The executor does not call task done on processed queue. Therefore, cannot join.
            # self._processed_frames_queue.join()
            # self._active_model_queue.join()
            pass
        else:
            pass

    def _start_threads(self):
        if _EXECUTOR_TYPE == "thread":
            self._executor_thread = threading.Thread(target=_executor,
                                                     args=(self.opts.simulate_gpu, self._gpu_number, self._profiled_latencies,
                                                           self._processed_frames_queue, self._output_queue,
                                                           self._active_model_queue, self._close_event))
        else:
            self._executor_thread = mp.Process(target=_executor,
                                               args=(self.opts.simulate_gpu, self._gpu_number, self._profiled_latencies,
                                                     self._processed_frames_queue, self._output_queue,
                                                     self._active_model_queue, self._close_event))

        self._loader_thread = threading.Thread(target=self._loader)
        self._executor_thread.start()
        self._loader_thread.start()

    def _load_all_models_cpu(self):
        '''Load model pool from the disk into cpu memory.'''
        model_utils.set_deterministic_behaviour(seed=1)
        model_utils.set_cuda_device(self._gpu_number)
        models = [model_utils.generate_model(self.opts, model_number=i, use_gpu=False).share_memory()
                  for i in range(self.opts.n_models)]
        return models

    def _load_init_models_cuda(self):
        ''' Load active model on the GPU '''
        self._active_model_number = self.opts.init_model_number
        self._active_batch_size = self.opts.init_batch_size
        half_count = self.opts.active_model_count // 2
        start_model = self._model_lb(self._active_model_number - half_count)
        end_model = self._model_ub(self._active_model_number + half_count)
        for model in range(start_model, end_model + 1):
            self._models[model].cuda(self._gpu_number)
        torch.cuda.synchronize(self._gpu_number)

        # Put in the active model queue so that executor can fetch it.
        self._active_model_queue.put((self._active_model_number,
                                      self._active_batch_size,
                                      self._models[self._active_model_number]))

    def _model_lb(self, x):
        if x < 0:
            x = 0
        return x

    def _model_ub(self, x):
        if x >= self.opts.n_models:
            x = self.opts.n_models - 1
        return x

    def run(self):
        ''' Use the main thread for preprocessing. '''
        def shape_incoming_requests(cycle_start, pushed_req_count):
            with self._lock:
                active_model_number = self._active_model_number
                active_batch_size = self._active_batch_size
            active_duty_cycle = self._profiled_latencies[active_model_number][active_batch_size-1]
            interval = (time.time() - cycle_start) * 1e3
            reset_flag = False
            if ((active_duty_cycle - interval) > 1.0):
                if pushed_req_count >= active_batch_size:
                    print(f"Cannot push this request, sleeping for {(active_duty_cycle - interval)}, "
                          f" interval {interval}, active_duty_cycle {active_duty_cycle}")
                    utils.sleepInMillis((active_duty_cycle - interval))
                    reset_flag = True
                else:
                    print(
                        f"Pushing due to more batch handling, {pushed_req_count}/{active_batch_size}")
            else:
                print(
                    f"Pushing due to new duty cycle {active_model_number}, {interval}")
                reset_flag = True

            if reset_flag:
                pushed_req_count = 0
                cycle_start = time.time()
            return cycle_start, pushed_req_count

        self._init_process()
        logging.info(f"Worker main thread on gpu {self._gpu_number} started!")
        cycle_start = time.time()
        pushed_req_count = 0
        while not self._close_event.is_set():
            try:
                res = self._input_queue.get()
                if res is None:
                    continue

                metadata, raw_frame = res
                metadata.worker_recv_ts = time.time()
                # TODO: Decide where to perform unserialization.
                # and also use thread pool to parallelize the process
                # processed_frame = self.unserialize(raw_frame)
                processed_frame = raw_frame
                # cycle_start, pushed_req_count = shape_incoming_requests(
                #     cycle_start, pushed_req_count)
                metadata.worker_processing_ts = time.time()
                logging.debug(f"Received frame {metadata.frame_id} from "
                              f"client {metadata.client_id} on GPU {self._gpu_number} at"
                              f" ts {utils.timestamp_to_str(metadata.worker_recv_ts)} and pushed at"
                              f" ts {utils.timestamp_to_str(metadata.worker_processing_ts)}")
                self._processed_frames_queue.put((metadata, processed_frame))
                pushed_req_count += 1
            except queue.Empty:
                pass
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                logging.warn(f"Exception {e}")
        self._processed_frames_queue.put(None)
        self._stop_process()
        logging.info(f"Worker main thread on gpu {self._gpu_number} stopped!")

    def set_desired_model(self, desired_model_number, desired_batch_size):
        # TODO: Avoid communicating to the worker,
        # if there is no change in the model number or the batch size
        self._control_input_queue.put(
            (desired_model_number, desired_batch_size))

    def _loader(self):
        ''' The thread to (un)load models on GPUs '''
        def get_models_set():
            half_count = self.opts.active_model_count // 2
            active_models = set(range(self._model_lb(active_model_number - half_count),
                                      self._model_ub(active_model_number+half_count)+1))
            desired_models = set(range(self._model_lb(desired_model_number-half_count),
                                       self._model_ub(desired_model_number + half_count)+1))

            models_intersection = active_models.intersection(desired_models)
            logging.debug(f"Active model count: {self.opts.active_model_count}, "
                          f"Active models: {active_models}, DesiredModels:"
                          f"{desired_models}, Intersection: {models_intersection}")
            to_unload_set = active_models - models_intersection

            to_load_set = desired_models - models_intersection
            to_load_set.discard(desired_model_number)  # Explicitly load it

            model_cache_hit = False
            if desired_model_number in active_models:
                model_cache_hit = True

            return to_unload_set, to_load_set, model_cache_hit

        def log_timings(start_time, t1, t2, t3, t4, end_time):
            fmt_str = "\n" + '{:=^100}'.format('Loader Timings') + "\n"
            fmt_str += "\t{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}".format("TotalTime",
                                                                      "ModelsUnLoadTime",
                                                                      "ActiveModelLoadTime",
                                                                      "ActiveModelUnLoadTime",
                                                                      "ModelsLoadTime") + "\n"
            fmt_str += "\t{:<20.2f}{:<20.2f}{:<20.2f}{:<20.2f}{:<20.2f}"\
                "".format((end_time - start_time)*1e3, (t2-t1) * 1e3,
                          (t3-t2) * 1e3, (t4-t3) * 1e3, (end_time - t4)*1e3) + "\n"
            return fmt_str

        logging.info(f"Worker loader on gpu {self._gpu_number} started!")
        model_loading_count, model_miss_count = 0, 0
        cuda_stream = torch.cuda.Stream(self._gpu_number)
        while not self._close_event.is_set():
            # Receive control command to change the active model
            try:
                res = self._control_input_queue.get()
                if res is None:
                    continue
                desired_model_number, desired_batch_size = res
                logging.info(f"Loader: Got a model ({desired_model_number}, {desired_batch_size})"
                             f" change request on gpu {self._gpu_number}")
                assert desired_model_number >= 0 and desired_model_number < self.opts.n_models
                assert desired_batch_size > 0 and desired_batch_size < self.opts.max_batch_size
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                logging.warn(
                    f"Exception: getting control command to worker, {e}")

            load_flag = False
            with self._lock:
                active_model_number = self._active_model_number
                if desired_model_number != active_model_number:
                    load_flag = True
                elif desired_batch_size != self._active_batch_size:
                    self._active_batch_size = desired_batch_size
                    # We should inform the executor to start with new batch size
                    self._active_model_queue.put((self._active_model_number,
                                                  self._active_batch_size,
                                                  self._models[self._active_model_number]))

            if load_flag is False:
                utils.sleepInMillis(1)
                continue

            try:
                model_loading_count += 1
                start_time = time.time()
                # Start loading process
                # Get the active and desired models set
                to_unload_set, to_load_set, model_cache_hit = get_models_set()

                with torch.cuda.stream(cuda_stream):
                    # # 1. Remove other active models from the GPUs
                    t1 = time.time()
                    # for model in to_unload_set:
                    #     self._models[model].cpu()

                    # 2. Load desired model
                    t2 = time.time()
                    if model_cache_hit is False:
                        self._models[desired_model_number].cuda(
                            self._gpu_number)
                        cuda_stream.synchronize()
                        model_miss_count += 1
                    with self._lock:
                        self._active_model_number = desired_model_number
                        self._active_batch_size = desired_batch_size

                    # Send the active model notification and wait to be picked up
                    self._active_model_queue.put((self._active_model_number,
                                                  self._active_batch_size,
                                                  self._models[self._active_model_number]))
                    self._active_model_queue.join()

                    # 3. Unload active the model.
                    # t3 = time.time()
                    # self._models[active_model_number].cpu()

                    t3 = time.time()
                    for model in to_unload_set:
                        self._models[model].cpu()

                    # 4. Load remaining desired models to the GPUs
                    t4 = time.time()
                    for model in to_load_set:
                        self._models[model].cuda(self._gpu_number)

                cuda_stream.synchronize()
                end_time = time.time()
                logging.debug(log_timings(
                    start_time, t1, t2, t3, t4, end_time))

                model_hit_count = model_loading_count - model_miss_count
                logging.debug(f"Loader stats: model loading count = {model_loading_count}, "
                              f"model miss count = {model_miss_count}, model hit count = {model_hit_count}")
            except Exception as e:
                logging.warn(f"Worker Loader: {str(e)}")

        logging.info(f"Worker loader on gpu {self._gpu_number} stopped!")
