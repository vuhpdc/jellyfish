import logging
from src.server.controller.utils import printModelIDsOnGpu
from src.server.controller.utils import print_info_str
from src.utils import setup_logging
import multiprocessing as mp
from src.utils import Logger
import os
import time


class Scheduler(mp.Process):
    def __init__(self, opts, env_data, initial_models_info) -> None:
        super(Scheduler, self).__init__()
        self.opts = opts
        self._prev_models_info = initial_models_info
        self._env_data = env_data

        # Things declared here will be shared with the parent process
        self._input_queue = mp.SimpleQueue()
        self._output_queue = mp.SimpleQueue()

        # alloc events to signal
        self._close_event = mp.Event()

    def close(self):
        self._close_event.set()
        self._input_queue.put(None)

    def map_clients(self, clients_info):
        self._input_queue.put(clients_info)
        mapping = self._output_queue.get()
        return mapping.clients_map, mapping.models_info

    def _init_process(self):
        # Setup logging
        setup_logging(self.opts, log_name="scheduler", run_mode="RELEASE")
        # setup_logging(self.opts, log_name="scheduler", run_mode="DEBUG")
        self._select_algos()
        self._scheduler_run_count = 0
        self._stats = Logger(os.path.join(self.opts.log_path, "scheduler.csv"),
                             ['scheduler_run_count', 'old_models_info',
                              'new_models_info', 'excution_time'])

    def _select_algos(self):
        def model_selection_algo():
            if self.opts.selection_algo == "SA":
                from src.server.controller.selection_algo import ModelSelectionSimulatedAnnealing as selection_algo
            else:
                raise RuntimeError("Not Implemented!")
            return selection_algo

        def client_mapping_algo():
            if self.opts.client_mapping_algo == "DP":
                from src.server.controller.mapping_algo import DP_on_AggregateRate as mapping_algo
            else:
                raise RuntimeError("Not Implemented!")
            return mapping_algo

        self._model_selection_func = model_selection_algo()
        self._client_mapping_func = client_mapping_algo()

    @staticmethod
    def _clear_models_info(models_info):
        for _, model in models_info.items():
            model.reset()

    def run(self):
        ''' runs the scheduler algo whenever it
        receives command from the control manager '''
        self._init_process()
        logging.info("started!")
        while not self._close_event.is_set():
            try:
                client_info = self._input_queue.get()
                if client_info is None:
                    continue
                logging.debug("Request to map")
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                logging.error(
                    f"Exception in receiving control command: {e}")

            self._scheduler_run_count += 1
            self._clear_models_info(self._prev_models_info)

            logging.debug(print_info_str(client_info, self._prev_models_info))
            old_models_info = printModelIDsOnGpu(self._prev_models_info)
            start_time = time.time()
            mapping_info, _ = self._model_selection_func(simData=self._env_data,
                                                         initialModelsInfo=self._prev_models_info,
                                                         clientsInfo=client_info,
                                                         MappingAlgo=self._client_mapping_func,
                                                         shuffleModels=True)
            execution_time = (time.time() - start_time) * 1e3
            self._prev_models_info = mapping_info.models_info

            logging.debug(mapping_info.print_str())
            logging.debug(mapping_info.metrics.print_str())
            new_models_info = printModelIDsOnGpu(self._prev_models_info)

            self._stats.log({'scheduler_run_count': self._scheduler_run_count,
                             'old_models_info': old_models_info,
                             'new_models_info': new_models_info,
                             'excution_time': round(execution_time, 3)})
            self._output_queue.put(mapping_info)

        logging.info("stopped!")
