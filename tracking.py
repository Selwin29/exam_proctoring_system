from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100.0):
        # next unique object ID
        self.next_object_id = 1
        
        # OrderedDicts to keep track of mapped objects
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects is a list of [x1, y1, x2, y2]
        
        if len(rects) == 0:
            # no objects detected, mark everyone as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy()

        # initialize array of input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute distance between each pair of object centroids and input centroids
            # Using scipy.spatial.distance.cdist is optimal, doing manual numpy to avoid dependency
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # sort row to find min distance matching
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check distance constraint
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Check for disappeared objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Check for new objects
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects.copy()

class FaceTracker:
    def __init__(self, cfg):
        self.tracker = CentroidTracker(
            max_disappeared=cfg.max_disappeared, 
            max_distance=cfg.max_distance
        )
        self.primary_id = None
        self.primary_lost_frames = 0
    
    def process_faces(self, face_state, frame_w, frame_h):
        """
        Takes FaceState with all_bboxes and returns a mapping of face IDs
        to bounding boxes, plus a proxy alert if the main ID swaps.
        """
        rects = []
        if face_state.detected and len(face_state.all_bboxes) > 0:
            for bbox in face_state.all_bboxes:
                x1n, y1n, x2n, y2n = bbox
                x1, y1 = int(x1n * frame_w), int(y1n * frame_h)
                x2, y2 = int(x2n * frame_w), int(y2n * frame_h)
                rects.append([x1, y1, x2, y2])
        
        objects = self.tracker.update(rects)
        
        # Track our primary tracked candidate
        if len(objects) == 1 and self.primary_id is None:
            # First face seen becomes primary applicant
            self.primary_id = list(objects.keys())[0]
            
        proxy_alert = False
        if self.primary_id is not None:
            if self.primary_id not in objects:
                self.primary_lost_frames += 1
            else:
                self.primary_lost_frames = 0

            # If someone is there, but primary ID is not, it means an ID swap
            if len(objects) > 0 and self.primary_id not in objects:
                proxy_alert = True
                
        return objects, proxy_alert
