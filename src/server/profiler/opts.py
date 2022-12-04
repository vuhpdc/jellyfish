from src.server.opts import argument_parser as server_argument_parser


def argument_parser():
    parser = server_argument_parser()

    parser.add_argument('--gt_annotations_path', type=str,
                        default='instances_val2017.json', help='ground truth annotations file')
    parser.add_argument('--dataset_dir', type=str,
                        default=None, help='dataset dir')
    parser.add_argument('--total_profile_iter', type=int,
                        default=1000, help='Total iterations')
    parser.add_argument('--profile_dir', type=str, default="", help="Log dir")

    return parser


def parse_args():
    parser = argument_parser()
    args = parser.parse_args()
    return args
