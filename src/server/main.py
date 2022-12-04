from src.server.request_dispatcher import RequestDispatcher
from src.server.controller.manager import ControlManager
from src.server.worker.manager import WorkerManager
from src.server.model_server import ModelServer
from src.server.opts import parse_args
import signal
import sys
from src.utils import read_profiled_latencies, read_profiled_accuracies, setup_logging
import time
import gc
gc.disable()


def signal_handler(signum, frame):
    server.stop()
    request_dispatcher.close()
    control_manager.close()
    worker_manager.close()
    sys.exit(0)


if __name__ == '__main__':
    # Parse command line arguments
    opts = parse_args()

    # Setup logging
    setup_logging(opts, log_name="main_server")

    # Read profiled values
    profiled_latencies = read_profiled_latencies(
        opts.profiled_dir, opts.n_models, opts.max_batch_size)
    profiled_accuracies = read_profiled_accuracies(
        opts.profiled_dir, opts.n_models)

    # Start control manager, workers and dispatcher
    worker_manager = WorkerManager(opts, profiled_latencies)
    control_manager = ControlManager(opts, profiled_latencies,
                                     profiled_accuracies, worker_manager)

    request_dispatcher = RequestDispatcher(
        opts, worker_manager, control_manager)

    # Setup signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # TODO: Wait is introduced for processes to setup completely.
    # Replace with proper status check
    time.sleep(10)
    # Start model server
    server = ModelServer(request_dispatcher,
                         opts.server_port, opts.grpc_max_workers)
    server.start_and_wait()
