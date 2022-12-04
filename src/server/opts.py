import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Moth Server")

    #---Server config---#
    parser.add_argument("--n_gpus", type=int,
                        default=1, help="Number of gpus")
    parser.add_argument("--server_port", type=int,
                        default=10001, help="Server port number")
    parser.add_argument("--grpc_max_workers", type=int,
                        default=40, help="Number of grpc workers")
    parser.add_argument("--run_mode", type=str, default="RELEASE",
                        choices=["DEBUG", "RELEASE"], help="Serving running mode")
    parser.add_argument("--log_path", type=str, default="./logs",
                        help="Path to save logs")

    #---Worker config---#
    parser.add_argument("--n_models", type=int,
                        default=19, help="Total number of models")
    parser.add_argument("--init_model_number", type=int,
                        default=0, help="Initial model to load")
    parser.add_argument("--init_batch_size", type=int,
                        default=1, help="Initial batch size")
    parser.add_argument("--active_model_count", type=int,
                        default=5, help="Total models loaded on GPU at any time")
    parser.add_argument("--max_batch_size", type=int,
                        default=12, help="Maximum batch size to support")
    parser.add_argument('--weights_dir', type=str,
                        help='Directory containing pytorch weights')
    parser.add_argument('--model_config_dir', type=str,
                        default='cfg/', help='Model config directory')
    parser.add_argument('--simulate_gpu', action='store_true',
                        help="Use simulated pytorch models")
    parser.set_defaults(simulate_gpu=False)

    #---Controller config---#
    parser.add_argument('--schedule_interval', type=float, default=10,
                        help='Periodic interval (sec) to run scheduler')
    parser.add_argument('--schedule_min_interval', type=float, default=10,
                        help='Minimum interval between consecutive scheduler runs')
    parser.add_argument('--profiled_dir', type=str,
                        help='Directory of profiled values')
    parser.add_argument('--fps_lcd', type=int, default=10,
                        help='LCD of clients FPS')
    parser.add_argument('--effectiveness_threshold', type=int, default=1.0,
                        help='Effective number of clients to support')
    parser.add_argument('--selection_algo', type=str, default="SA",
                        help='Model selection scheduler algorithm')
    parser.add_argument('--client_mapping_algo', type=str, default="DP",
                        help='Client mapping scheduler algorithm')

    return parser


def parse_args():
    parser = argument_parser()
    args = parser.parse_args()
    return args
