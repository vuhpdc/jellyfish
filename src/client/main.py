import logging
from signal import SIGTERM
import subprocess
import time
from src.client.model_serving_client import ModelServingClient as ModelClient
from src.client.frame_manager import FrameReader
from src.client.controller_client import ControllerClient
from src.client.response_manager import ResponseHandler
from src.client.opts import parse_args
from src.client.fps import FPSLogger
from src.utils import Logger, setup_logging
import os

metadata_fields = ['client_id', 'frame_id', 'orig_image_height', 'orig_image_width',
                   'frame_wire_size', 'desired_model', 'used_model', 'next_model',
                   'client_send_req_ts', 'client_recv_res_ts', 'client_recv_ack_ts',
                   'server_recv_req_ts', 'server_send_res_ts', 'server_send_ack_ts',
                   'network_time', 'bw']


class FrameMetadata(object):
    __slots__ = metadata_fields

    def __init__(self) -> None:
        self.client_id = -1
        self.frame_id = 0
        self.orig_image_height = 0
        self.orig_image_width = 0
        self.frame_wire_size = 0
        self.desired_model = -1
        self.used_model = -1
        self.next_model = -1
        self.client_send_req_ts = 0.0
        self.client_recv_res_ts = 0.0
        self.server_recv_req_ts = 0.0
        self.server_send_res_ts = 0.0
        self.server_send_ack_ts = 0.0
        self.client_recv_ack_ts = 0.0
        self.network_time = 0.0
        self.bw = 0.0


def start_bw_shaping(opts):
    if opts.shaping_script == "":
        return None

    args = ["/bin/bash", opts.shaping_script, opts.shaping_data_file]
    proc = subprocess.Popen(args)

    return proc


def stop_bw_shaping(shaping_proc):
    if shaping_proc == None:
        return
    shaping_proc.send_signal(SIGTERM)
    shaping_proc.wait()


if __name__ == "__main__":
    opts = parse_args()

    # Setup loggin
    setup_logging(opts, "client")

    # Frame reader
    frame_reader = FrameReader(
        video_file=opts.video_file, image_list=opts.image_list, frame_rate=opts.frame_rate)

    # Loggers
    fps_logger = FPSLogger(log_path=opts.log_path)
    stats_logger = Logger(os.path.join(opts.log_path, opts.stats_fname),
                          ['client_id', 'frame_id', 'send_ts', 'recv_ts', 'dropped_frame',
                           'e2e_latency', 'desired_model', 'used_model', 'next_model',
                           'network_time', 'frame_wire_size'])

    # Stub to controller
    controller_client = ControllerClient(host=opts.controller_host,
                                         port_number=opts.controller_port,
                                         frame_rate=opts.frame_rate,
                                         slo=opts.slo,
                                         init_bw=opts.init_bw,
                                         lat_wire=opts.lat_wire)

    # Server response handler
    output_manager = ResponseHandler(opts=opts,
                                     controller_client=controller_client,
                                     fps_logger=fps_logger,
                                     stats_logger=stats_logger,
                                     log_path=opts.log_path)

    # Stub to server
    model_client = ModelClient(
        host=opts.model_server_host, port_number=opts.model_server_port, callback=output_manager, stats=False)

    client_id, init_model_number = model_client.register_with_server(slo=opts.slo,
                                                                     frame_rate=opts.frame_rate,
                                                                     lat_wire=opts.lat_wire,
                                                                     init_bw=controller_client.get_bw())
    controller_client.update_next_model(init_model_number)

    # Start bw shaping, if yes
    # shaping_proc = start_bw_shaping(opts)

    fps_logger.start()
    fps_logger.start_logging()
    prev_time = time.time()
    while True:
        frame = frame_reader.next()
        if frame is None:
            break

        # Frame reading stats
        cur_time = time.time()
        logging.debug(
            f"Frame {frame['id']} reading interval: {(cur_time - prev_time)*1e3}")

        prev_time = cur_time

        # Metadata for the frame
        metadata = FrameMetadata()
        metadata.client_id = client_id
        metadata.frame_id = frame["id"]
        metadata.desired_model = controller_client.get_next_model()
        metadata.bw = controller_client.get_bw()
        metadata.orig_image_height, metadata.orig_image_width = frame["data"].shape[:2]

        model_client.predict(metadata, frame["data"])
        output_manager.add_outstanding_req(metadata)

    frame_reader.close()
    fps_logger.close()
    fps_logger.join()
    model_client.unregister_with_server()
    output_manager.close()

    # stop_bw_shaping(shaping_proc)

    print("Done!")
