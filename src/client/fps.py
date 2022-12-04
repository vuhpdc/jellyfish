import threading
import time
from src.utils import Logger
import os


class FPSLogger(threading.Thread):
    """
    Periodically logs number of frames processed.
    """

    def __init__(self, timer=1, log_path="", log_name='fps'):
        threading.Thread.__init__(self)
        self._num_frames = 0
        self._fps_list = []
        self._lock = threading.Lock()
        self.timer = timer
        self.logger = Logger(os.path.join(log_path, log_name + ".log"),
                             ['FPS'])
        self._STARTED = False
        self._TERMINATE = False
        self._warmup_count = 10

    def run(self):
        while True:
            with self._lock:
                if self._TERMINATE:
                    break
                if self._STARTED == True:
                    fps = self._num_frames / self.timer
                    self._fps_list .append(fps)
                    self._num_frames = 0
            time.sleep(self.timer)

    def close(self):
        self.stop_logging()
        self.save()
        with self._lock:
            self._TERMINATE = True

    def is_started(self):
        with self._lock:
            return self._STARTED

    def start_logging(self):
        self._num_frames = 0
        self._fps_list.clear()
        with self._lock:
            self._STARTED = True

    def stop_logging(self):
        with self._lock:
            self._STARTED = False

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        with self._lock:
            self._num_frames += 1

    def save(self):
        with self._lock:
            self._fps_list = self._fps_list[self._warmup_count:-1]
            for fps in self._fps_list:
                self.logger.log({'FPS': fps})
