import time
import queue
import threading

class DetectorThread(threading.Thread):
    def __init__(self, in_queue: queue.Queue, out_queue: queue.Queue, 
                 face_monitor, pose_estimator, phone_detector, 
                 yolo_every_n_frames: int = 3):
        """
        Dedicated thread for ML inference.
        Runs MediaPipe Face detection on every frame.
        Runs YOLO phone detection every N frames to save compute.
        """
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.face_monitor = face_monitor
        self.pose_estimator = pose_estimator
        self.phone_detector = phone_detector
        self.yolo_every_n_frames = yolo_every_n_frames
        
        self.running = False

    def run(self):
        print("[DetectorThread] Thread started. Waiting for frames...")
        
        # Import to dummy initialize the result
        from phone_detector import PhoneDetectionResult
        last_phone_result = PhoneDetectionResult()
        
        self.running = True
        while self.running:
            try:
                # Block until a frame is available
                data = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            frame_count, raw_frame, proc_frame, rgb_frame = data
            
            start_t = time.time()
            
            # --- FACE MESH (Every frame) ---
            face_state = self.face_monitor.process(rgb_frame)
            
            # --- HEAD POSE (Every frame) ---
            pose = self.pose_estimator.estimate(face_state)
            
            # --- YOLO DETECTOR (Every N frames) ---
            if frame_count % self.yolo_every_n_frames == 0:
                last_phone_result = self.phone_detector.process(proc_frame, face_state.centroid_px)
            
            phone_result = last_phone_result
            
            proc_time = time.time() - start_t
            
            # Package the results
            result_data = (
                frame_count,
                raw_frame,
                proc_frame,
                face_state,
                pose,
                phone_result,
                proc_time
            )
            
            # Push to the out queue
            try:
                self.out_queue.put_nowait(result_data)
            except queue.Full:
                try:
                    self.out_queue.get_nowait() # discard oldest
                    self.out_queue.put_nowait(result_data)
                except queue.Empty:
                    pass

    def stop(self):
        self.running = False
