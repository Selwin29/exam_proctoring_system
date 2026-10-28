import cv2
import time
import queue
import threading
from typing import Tuple

class CameraThread(threading.Thread):
    def __init__(self, device_index: int, capture_res: Tuple[int, int], proc_res: Tuple[int, int], fps: int, buffer_size: int, out_queue: queue.Queue):
        """
        Dedicated thread for capturing frames from the camera.
        Drops frames if the out_queue is full to ensure we always process the most recent realtime frame.
        """
        super().__init__(daemon=True)
        self.device_index = device_index
        self.capture_w, self.capture_h = capture_res
        self.proc_w, self.proc_h = proc_res
        self.fps = fps
        self.buffer_size = buffer_size
        self.out_queue = out_queue
        
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.device_index}")

        # Set hardware properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        self.running = False
        self.frame_count = 0

    def run(self):
        self.running = True
        while self.running:
            ret, raw_frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.frame_count += 1
            
            # Preprocess frame before queuing to save detector thread time
            proc_frame = cv2.resize(raw_frame, (self.proc_w, self.proc_h))
            rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

            # Package the data tuple
            data = (self.frame_count, raw_frame, proc_frame, rgb_frame)

            try:
                # If queue is full, we remove the oldest frame so we don't lag behind reality
                self.out_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.out_queue.get_nowait() # discard oldest
                    self.out_queue.put_nowait(data) # put newest
                except queue.Empty:
                    pass

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
