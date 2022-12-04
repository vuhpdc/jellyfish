from src.server.worker.worker import Worker
import logging


class WorkerManager(object):
    def __init__(self, opts, profiled_latencies) -> None:
        self.num_gpus = opts.n_gpus
        self.profiled_latencies = profiled_latencies
        self._workers_lst = [None] * self.num_gpus
        self._start_workers(opts)

    def _start_workers(self, opts):
        for i in range(self.num_gpus):
            worker = Worker(opts, gpu_number=i,
                            profiled_latencies=self.profiled_latencies)
            worker.start()
            self._workers_lst[i] = worker
            logging.info(f"Worker started on GPU {i}")

    def _stop_workers(self):
        for i in range(self.num_gpus):
            self._workers_lst[i].close()
            self._workers_lst[i].join()

    def close(self):
        self._stop_workers()

    def update_worker_model(self, gpu_number, model_number, batch_size):
        self._workers_lst[gpu_number].set_desired_model(
            model_number, batch_size)

    def schedule(self, gpu_number, request):
        self._workers_lst[gpu_number].put_request(request)

    def get_response(self, gpu_number, non_block=True):
        if non_block:
            response = self._workers_lst[gpu_number].get_response_nowait()
        else:
            raise RuntimeError("Not implemented!")
        return response
