import json
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

log_file = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\logs\proctoring.log"
graphs_dir = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\graphs"
os.makedirs(graphs_dir, exist_ok=True)

# Data structures
suspicion_times = []
suspicion_scores = []

pitch_times = []
pitch_angles = []

yaw_times = []
yaw_angles = []

absence_times = []
face_absent_durations = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            parts = line.split(" | ", 2)
            if len(parts) == 3:
                date_str = parts[0]
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                
                payload = json.loads(parts[2])
                event_type = payload.get("event")
                data = payload.get("data", {})
                msg = payload.get("msg") or payload.get("message") or ""
                
                # 1. Suspicion/Cheating Scores
                if event_type == "suspicion":
                    match = re.search(r"Score: ([\d\.]+)", msg)
                    if match:
                        suspicion_scores.append(float(match.group(1)))
                        suspicion_times.append(dt)
                elif event_type == "cheating_alert" and "score" in data:
                    suspicion_scores.append(float(data["score"]))
                    suspicion_times.append(dt)
                
                # 2. Pitch/Yaw
                if "pitch" in data:
                    pitch_angles.append(float(data["pitch"]))
                    pitch_times.append(dt)
                if "yaw" in data:
                    yaw_angles.append(float(data["yaw"]))
                    yaw_times.append(dt)
                
                # 3. Face absence duration
                if event_type == "face_absent":
                    match = re.search(r"([\d\.]+)s", msg)
                    if match:
                        dur = float(match.group(1))
                        if dur > 0.0:
                            face_absent_durations.append(dur)
                            absence_times.append(dt)
                            
        except Exception:
            pass

plt.style.use("ggplot")

# Graph 1: Suspicion Score over Time (Scatter/Line)
plt.figure(figsize=(10, 6))
plt.plot(suspicion_times, suspicion_scores, marker='o', linestyle='-', color='crimson', markersize=4, alpha=0.8)
plt.title("Suspicion/Cheating Alerts Over Time (Line Plot)", pad=15)
plt.xlabel("Time")
plt.ylabel("Severity Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "suspicion_scores1.png"), dpi=300)
plt.close()

# Graph 2: Pitch Angle Timeline (Scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(pitch_times, pitch_angles, color='blue', alpha=0.6, s=30)
plt.title("Vertical Head Pitch Angles Over Time (Scatter Plot)", pad=15)
plt.xlabel("Time")
plt.ylabel("Pitch (Degrees) - Negative is down")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "head_pitch_timeline2.png"), dpi=300)
plt.close()

# Graph 3: Yaw Angle Timeline (Scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(yaw_times, yaw_angles, color='purple', alpha=0.6, s=30)
plt.title("Horizontal Head Yaw Angles Over Time (Scatter Plot)", pad=15)
plt.xlabel("Time")
plt.ylabel("Yaw (Degrees) - Deviations from center")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "head_yaw_timeline3.png"), dpi=300)
plt.close()

# Graph 4: Face Absence Durations Over Time (Stem plot)
plt.figure(figsize=(10, 6))
plt.stem(absence_times, face_absent_durations, linefmt='orange', markerfmt='D', basefmt=" ", use_line_collection=True)
plt.title("Duration of Face Absence Events Over Time (Stem Plot)", pad=15)
plt.xlabel("Time")
plt.ylabel("Duration (Seconds)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "face_absence_timeline4.png"), dpi=300)
plt.close()

print("All graphs successfully plotted using alternative visuals.")
