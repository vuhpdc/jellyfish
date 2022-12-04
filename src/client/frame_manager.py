import cv2
import threading
from collections import deque
import time


def getCurTime():
    cur_time = time.time()
    return cur_time


class ImageReader(object):
    def __init__(self, image_list, frame_rate) -> None:
        # self.image_list = image_list
        self.frame_rate = frame_rate
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._run)
        self._close_event = threading.Event()
        self.images_path = self.read_image_list(image_list)
        self.total_images = len(self.images_path)
        self.reset()

        self._reader.start()

    def close(self):
        self._close_event.set()
        self._reader.join()

    def reset(self):
        self.loaded_images = deque([], maxlen=100)
        self._image_idx = 0

    @staticmethod
    def read_image_list(image_list):
        with open(image_list) as file:
            images_path = file.readlines()
        images_path = [path.strip() for path in images_path]
        return images_path

    def _run(self):
        while not self._close_event.is_set():
            with self._lock:
                if self._image_idx < self.total_images and \
                        len(self.loaded_images) < 100:
                    img = cv2.imread(self.images_path[self._image_idx])
                    self.loaded_images.append(img)
                    self._image_idx += 1

    def next(self):
        img = None
        with self._lock:
            if self._image_idx < self.total_images or \
                    len(self.loaded_images) > 0:
                img = self.loaded_images.pop()

        return img


class VideoReader(object):
    def __init__(self, video_file, frame_rate) -> None:
        self.video_file = video_file
        self.frame_rate = frame_rate
        self._cap = cv2.VideoCapture(video_file)

    def close(self):
        if self._cap is not None:
            self._cap.release()

    def next(self):
        ret, frame = self._cap.read()
        if ret is False:
            return None
        return frame


class FrameReader(object):
    def __init__(self, video_file, image_list, frame_rate) -> None:
        self.video_file = video_file
        self.image_list = image_list
        self.frame_rate = frame_rate
        self.frame_interval = (1.0 / frame_rate) * 1e3
        if video_file != "":
            self._reader = VideoReader(video_file, frame_rate)
        else:
            if image_list == "":
                raise ValueError("Missing image file for reading")
            self._reader = ImageReader(image_list, frame_rate)

        self.frame_id = 0
        self._last_read_time = getCurTime() - (self.frame_interval * 1e-3)

    def close(self):
        self._reader.close()

    def next(self):
        img = self._reader.next()
        if img is None:
            return None

        # Need to sleep
        time_diff_ms = (getCurTime() - self._last_read_time) * 1e3
        if (time_diff_ms < self.frame_interval):
            sleep_time_sec = (self.frame_interval - time_diff_ms) * 1e-3
            time.sleep(sleep_time_sec)

        frame = {
            "id": self.frame_id,
            "data": img
        }
        self.frame_id = self.frame_id + 1
        self._last_read_time = getCurTime()
        return frame
