import pandas as pd
import os
import numpy as np
import time
import csv
import logging
import sys

FRAME_SIZES = [128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]


def sleepInMillis(x):
    time.sleep(x*1e-3)


def bytes_to_kbits(byte_size):
    return (byte_size * 8) / 1024


def timestamp_to_str(ts):
    time_ms = round((ts * 1e3) % 1e3)
    str = time.strftime('%T.{}'.format(time_ms), time.localtime(ts))
    return str


def set_process_nice(nice_value=0):
    my_pid = os.getpid()
    os.system(f"sudo renice -n {nice_value} -p {my_pid}")


def read_profiled_accuracies(profiled_dir, total_models):
    file = os.path.join(profiled_dir, 'accuracy.txt')
    df = pd.read_csv(file, delimiter=",", header=0)
    accuracies = []
    for model_number in range(total_models):
        acc = df.loc[(df["ModelSize"] == FRAME_SIZES[model_number]),
                     "Accuracy"].values
        accuracies.append(acc[0])
    return np.array(accuracies)


def latency_estimator(latencies):
    '''
    Adjust latency measurements. We adjust it because of the measurement noise. We realized that at small batch sizes i.e., 1 or 2,
    Some models have lower latency, within (1-2ms) than the smaller model. We currently attribute this variation to the 
    measurement noise. Generally, our scheduling method assumes that the bigger model have larger latency and ideally it should be
    true. If you do not intend to adjust the curves, then you can adjust the percentile factor to calculate latency.

    You can also use the quantile regression to estimate the P99 latency, as specified in [Clipper, NSDI'17]
    '''
    for model_idx in range(latencies.shape[0]):
        for batch in range(latencies.shape[1]):
            if batch > 0 and latencies[model_idx][batch] < latencies[model_idx][batch-1]:
                correction = latencies[model_idx][batch -
                                                  1] - latencies[model_idx][batch]
                # assert correction < 2.00
                latencies[model_idx][batch] = latencies[model_idx][batch-1]
                logging.info(
                    f"Fixed profiled latencies for same model:{model_idx}, {batch}, {correction}")

            if model_idx > 0 and latencies[model_idx][batch] < latencies[model_idx-1][batch]:
                correction = latencies[model_idx -
                                       1][batch] - latencies[model_idx][batch]
                # assert correction < 2.00
                latencies[model_idx][batch] = latencies[model_idx-1][batch]
                logging.info(
                    f"Fixed profiled latencies for same batch:{model_idx}, {batch} {correction}")

    return latencies


def read_profiled_latencies(profiled_dir, total_models, max_batch_size):
    _PERCENTILE = 0.99
    _SLOWDOWN_FACTOR = 1.0
    latencies = np.zeros((total_models, max_batch_size), dtype=float)
    for model_number in range(total_models):
        file = os.path.join(profiled_dir, 'latency/profile_latency_{}.txt'.format(
            FRAME_SIZES[model_number]))
        df = pd.read_csv(file, delimiter=",", header=0)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        for batch in range(1, max_batch_size+1):
            df_batch = df[df["Batch"] == batch]
            samples = df_batch["InferenceTime"].iloc[0:]
            latencies[model_number, batch -
                      1] = (samples).quantile(_PERCENTILE) * _SLOWDOWN_FACTOR

    latencies = latency_estimator(latencies)
    return latencies


class AverageMeter(object):
    """Computes and stores the average and current value"""
    """Credit: Not sure who is the original author, if you know then update it """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """Credit: Not sure who is the original author, if you know then update it """

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter=',')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.log_file.flush()


def setup_logging(opts, log_name, run_mode="RELEASE"):
    class ProcessFilter(logging.Filter):
        def filter(self, rec):
            rec.process_name = log_name
            return True

    print(f"Setting up logger for {log_name} with level {run_mode}")
    debug_level = logging.DEBUG if run_mode == "DEBUG" else logging.WARN
    os.makedirs(opts.log_path, exist_ok=True)
    log_file = os.path.join(opts.log_path, f'{log_name}.log')

    logger = logging.getLogger()
    logger.handlers.clear()

    logging.basicConfig(level=debug_level,
                        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(process_name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(debug_level)
    formatter = logging.Formatter('%(process_name)s: %(message)s')
    console.setFormatter(formatter)

    logger.addHandler(console)
    # logger.addHandler(file_handler)
    logger.addFilter(ProcessFilter())
    logger.propagate = False
    return logger


class ErrorBasedFilter:
    '''
        Adaptive error-based filter proposed by, Kim et al. in "Mobile Network
        Estimation", MobiCom'2001
    '''

    def __init__(self, gamma=0.6, max_history=10) -> None:
        # define
        self._gamma = gamma
        self._MAX_HISTORY = max_history
        self._err_history_m = []
        self.reset()

    def reset(self):
        self._prev_est_m = -1
        self._prev_err_est_m = -1
        self._err_history_m = []
        self._err_idx_m = 0

    def update(self, O_t):
        if (self._prev_est_m == -1):
            self._prev_est_m = O_t
        else:
            # update error using EWMA
            self._update_err(O_t)

            # compute weight alpha
            err_max = max(self._err_history_m)
            alpha = 1.0 - (self._prev_err_est_m / err_max)

            # estimate value
            self._prev_est_m = alpha * \
                self._prev_est_m + (1.0 - alpha) * O_t

    def predict(self):
        return self._prev_est_m

    def _update_err(self, O_t):

        err = abs(self._prev_est_m - O_t)
        # estimate err
        if (self._prev_err_est_m == -1):
            self._prev_err_est_m = err
        else:
            self._prev_err_est_m = self._gamma * \
                self._prev_err_est_m + (1.0 - self._gamma) * err

        if len(self._err_history_m) == self._MAX_HISTORY:
            del self._err_history_m[0]
        self._err_history_m.append(self._prev_err_est_m)


def convert_cat_id_and_reorientate_bbox(single_annotation):
    '''
        Adjust category id according to COCO dataset annotations.
        Borrowed from: https://github.com/Tianxiaomo/pytorch-YOLOv4.git
    '''
    cat = single_annotation['category_id']
    bbox = single_annotation['bbox']
    x1, y1, w, h = bbox
    # x_center, y_center, w, h = bbox
    # x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    single_annotation['category_id'] = cat
    single_annotation['bbox'] = [x1, y1, w, h]  # COCO format
    return single_annotation


def compute_frame_size(total_models=15):
    # Currently, there's no way to know the encoded size of the image before encoding, as it depends
    # on the image content. We are relying on the minimum compression ratio seen in the (TEST) video.

    # DDS all videos min compression ratio, JPEG_ENCODE_90
    _COMPRESSION_RATIO = {128: 4.78, 160: 5.148, 192: 5.365, 224: 5.66, 256: 5.782, 288: 6.133,
                          320: 6.431, 352: 6.556, 384: 6.781, 416: 6.953, 448: 7.158, 480: 7.421,
                          512: 7.574, 544: 7.739, 576: 7.968, 608: 8.131, 640: 8.547}

    frame_sizes = np.zeros(total_models, dtype=float)
    for i, r in enumerate(FRAME_SIZES[:total_models]):
        frame_sizes[i] = ((r * r * 3 / 1024) /
                          _COMPRESSION_RATIO[r]) * 8  # Kbits

    return frame_sizes


def compute_net_throughput(size, total_time, wire_time, wire_time_conf=0.9):
    """
    :param size: Data size in bytes
    :param total_time: End2End transfer time in milliseconds
    :param wire_time: Propagation time on the wire in milliseconds
    :param wire_time_conf: Confidence in the wire_time value
    """
    transmission_time = total_time - (wire_time * wire_time_conf)

    # Wire time can be larger than the total time for two reasons.
    # 1. Clock offset issue between server and client, therefore, total_time is smaller.
    # 2. Incorrect wire_time estimation, i.e, estimated wire time is much larger
    # assert transmission_time > 0.000001, f"Wire time {wire_time} larger than total time {total_time}"
    logging.debug(f"Wire time {wire_time} and total time {total_time}")
    size_in_kbits = bytes_to_kbits(size)
    return (size_in_kbits / transmission_time) * 1000  # kbps


def compute_net_lat(size, bw, wire_time=0.0):
    """
    :param size: Data size in Kbits. It includes request, response sizes
    :param bw: Bandwidth in Kbps
    :param wire_time: One way propagation time on the wire in milliseconds
    :return: The total end2end delay, including round trip.
    """
    tranmission_time = (size / bw) * 1000
    total_time = tranmission_time + 2 * wire_time
    return total_time


def compute_total_lat(n, m):
    # End2End Latency function
    return n + 2*m


def compute_model_throughput(latency_mat):
    throughput_mat = []
    for _, lat_vec in enumerate(latency_mat):
        throughput_vec = []
        for i, lat in enumerate(lat_vec):

            t = ((i+1) / lat) * 1000
            throughput_vec.append(t)
        throughput_mat.append(throughput_vec)
    return throughput_mat
