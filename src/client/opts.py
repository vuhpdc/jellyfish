import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Moth Client")
    parser.add_argument("--video_file", type=str,
                        default="", help="Video file to stream")
    parser.add_argument("--image_list", type=str,
                        default="", help="File containint list of images")
    parser.add_argument("--frame_rate", type=int,
                        default=30, help="Frame rate")
    parser.add_argument("--slo", type=int,
                        default=100, help="Service level objective")
    parser.add_argument("--lat_wire", type=float,
                        default=0.5, help="Wire latency (ms)")
    parser.add_argument("--init_bw", type=int,
                        default=10000, help="Client bw in Kbps")
    parser.add_argument("--model_server_host", type=str,
                        default="localhost", help="Server host address")
    parser.add_argument("--model_server_port", type=int,
                        default=10001, help="Server port address")
    parser.add_argument("--controller_host", type=str,
                        default="localhost", help="Server host address")
    parser.add_argument("--controller_port", type=int,
                        default=9999, help="Server port address")
    parser.add_argument("--log_path", type=str, default="./logs",
                        help="Directory to store logs")
    parser.add_argument("--stats_fname", type=str, default="frame_stats.csv",
                        help="File name to save stats")
    parser.add_argument("--run_mode", type=str, default="RELEASE",
                        choices=["DEBUG", "RELEASE"], help="Client running mode")
    parser.add_argument("--shaping_script", type=str, default="",
                        help="Bandwidth shaping shell script")
    parser.add_argument("--shaping_data_file", type=str, default="",
                        help="File containing bw shaping values")

    args = parser.parse_args()

    return args
