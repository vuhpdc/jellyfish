import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description="MultiClient Simulation")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--num_models", type=int, default=16)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--fps_lcd", type=int, default=5)
    parser.add_argument("--effectiveness_threshold", type=float, default=1.0)
    parser.add_argument("--use_profiled_values",
                        action='store_true', help="Read data from profiles")
    parser.set_defaults(use_profiled_values=False)
    parser.add_argument('--profiled_dir', type=str, default="",
                        help='Directory of profiled values')

    args = parser.parse_args()
    args_dict = args.__dict__
    print('{:-^100}'.format('Configurations'))
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print('{:-^100}'.format(''))

    return args
