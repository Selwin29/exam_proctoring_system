import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from datetime import datetime
import re
import numpy as np

log_file = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\logs\proctoring.log"
out_dir = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\graphs"
os.makedirs(out_dir, exist_ok=True)

# Data structures
events_count = defaultdict(int)
events_by_date = defaultdict(lambda: defaultdict(int))
events_list = set()

suspicion_times = []
suspicion_scores = []

pitch_times = []
pitch_angles = []

yaw_times = []
yaw_angles = []

absence_times = []
face_absent_durations = []

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.split(" | ", 2)
                if len(parts) == 3:
                    date_str = parts[0]
                    # Parse date str
                    # Sometimes it's YYYY-MM-DD HH:MM:SS, let's extract date
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    day_str = date_str.split()[0]
                    
                    payload = json.loads(parts[2])
                    event_type = payload.get("event")
                    data = payload.get("data", {})
                    msg = payload.get("msg") or payload.get("message") or ""
                    
                    # Core Events
                    if msg:
                        msg_lower = msg.lower()
                        evt = None
                        if "phone" in msg_lower or event_type == "phone":
                            evt = "Phone Detected"
                        elif "down" in msg_lower or event_type == "look_down":
                            evt = "Looking Down"
                        elif "away" in msg_lower or "left" in msg_lower or "right" in msg_lower or event_type == "look_away":
                            evt = "Looking Away"
                        elif "multiple faces" in msg_lower or event_type == "multi_face":
                            evt = "Multiple Faces"
                        elif "absent" in msg_lower or event_type == "face_absent":
                            evt = "Face Absent"
                        elif event_type == "cheating_alert":
                            evt = "Cheating Alert"
                        
                        if evt:
                            events_count[evt] += 1
                            events_by_date[day_str][evt] += 1
                            events_list.add(evt)
                    
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

            except Exception as e:
                pass


plt.style.use("ggplot")

# ------------- Chart 1: Pie Chart -------------
if events_count:
    labels = list(events_count.keys())
    sizes = list(events_count.values())
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title("Distribution of Proctoring Violations (Pie Chart)", pad=20, fontsize=16)
    plt.axis('equal')
    out_path1 = os.path.join(out_dir, "name_of_graph1.png")
    plt.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.close()

    # ------------- Chart 2: Donut Chart -------------
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)
    plt.title("Distribution of Proctoring Violations (Donut Chart)", pad=20, fontsize=16)
    plt.axis('equal')
    out_path2 = os.path.join(out_dir, "name_of_graph2.png")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close()

    # ------------- Chart 3: Scatter Bubble Chart -------------
    plt.figure(figsize=(10, 6))
    x_positions = range(len(labels))
    plt.scatter(x_positions, sizes, color='purple', s=[s*100 for s in sizes], alpha=0.6)
    plt.xticks(x_positions, labels, rotation=45)
    plt.ylabel("Total Occurrences")
    plt.title("Occurrences by Violation Type (Scatter Bubble Chart)", pad=20, fontsize=16)
    plt.tight_layout()
    out_path3 = os.path.join(out_dir, "name_of_graph3.png")
    plt.savefig(out_path3, dpi=300, bbox_inches='tight')
    plt.close()

# ------------- Chart 4: Line Chart (Trend Over Time) -------------
if events_by_date:
    dates = sorted(events_by_date.keys())
    events_l = sorted(list(events_list))
    
    plt.figure(figsize=(10, 6))
    for evt in events_l:
        y_values = [events_by_date[d][evt] for d in dates]
        plt.plot(dates, y_values, marker='o', label=evt, linewidth=2)
    plt.title("Trend of Violations Over Time (Line Plot)", fontsize=16, pad=15)
    plt.ylabel("Occurrences")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    out_path4 = os.path.join(out_dir, "name_of_graph4.png")
    plt.savefig(out_path4, dpi=300, bbox_inches='tight')
    plt.close()

# ------------- Chart 5: Stacked Area Chart-------------
if events_by_date and len(dates) > 1:
    plt.figure(figsize=(10, 6))
    y_arrays = []
    labels_area = []
    for evt in events_l:
        y_arrays.append([events_by_date[d][evt] for d in dates])
        labels_area.append(evt)
    plt.stackplot(dates, y_arrays, labels=labels_area, alpha=0.7)
    plt.title("Cumulative Violations Trend (Stacked Area Chart)", fontsize=16, pad=15)
    plt.ylabel("Occurrences")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    out_path5 = os.path.join(out_dir, "name_of_graph5.png")
    plt.savefig(out_path5, dpi=300, bbox_inches='tight')
    plt.close()

# ------------- Chart 6: Suspicion Score (Scatter/Line) -------------
if suspicion_times:
    plt.figure(figsize=(10, 6))
    plt.plot(suspicion_times, suspicion_scores, marker='o', linestyle='-', color='crimson', markersize=4, alpha=0.8)
    plt.title("Suspicion Alerts Over Time (Line Plot)", pad=15)
    plt.xlabel("Time")
    plt.ylabel("Severity Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path6 = os.path.join(out_dir, "name_of_graph6.png")
    plt.savefig(out_path6, dpi=300)
    plt.close()

# ------------- Chart 7: Pitch Angles (Scatter) -------------
if pitch_times:
    plt.figure(figsize=(10, 6))
    plt.scatter(pitch_times, pitch_angles, color='blue', alpha=0.6, s=30)
    plt.title("Vertical Head Pitch Angles Over Time (Scatter Plot)", pad=15)
    plt.xlabel("Time")
    plt.ylabel("Pitch (Degrees)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path7 = os.path.join(out_dir, "name_of_graph7.png")
    plt.savefig(out_path7, dpi=300)
    plt.close()

# ------------- Chart 8: Yaw Angles (Scatter) -------------
if yaw_times:
    plt.figure(figsize=(10, 6))
    plt.scatter(yaw_times, yaw_angles, color='purple', alpha=0.6, s=30)
    plt.title("Horizontal Head Yaw Angles Over Time (Scatter Plot)", pad=15)
    plt.xlabel("Time")
    plt.ylabel("Yaw (Degrees)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path8 = os.path.join(out_dir, "name_of_graph8.png")
    plt.savefig(out_path8, dpi=300)
    plt.close()

# ------------- Chart 9: Face Absence Durations (Stem Plot) -------------
if absence_times:
    plt.figure(figsize=(10, 6))
    plt.stem(absence_times, face_absent_durations, linefmt='orange', markerfmt='D', basefmt=" ")
    plt.title("Face Absence Durations Over Time (Stem Plot)", pad=15)
    plt.xlabel("Time")
    plt.ylabel("Duration (Seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path9 = os.path.join(out_dir, "name_of_graph9.png")
    plt.savefig(out_path9, dpi=300)
    plt.close()

# ------------- Chart 10: Histogram of Suspicion Scores -------------
if suspicion_scores:
    plt.figure(figsize=(10, 6))
    plt.hist(suspicion_scores, bins=10, color='teal', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Suspicion Scores (Histogram)", pad=15)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    out_path10 = os.path.join(out_dir, "name_of_graph10.png")
    plt.savefig(out_path10, dpi=300)
    plt.close()

# ------------- Chart 11: Box Plot of Head Poses -------------
if pitch_angles or yaw_angles:
    plt.figure(figsize=(8, 6))
    data_to_plot = []
    labels_box = []
    if pitch_angles:
        data_to_plot.append(pitch_angles)
        labels_box.append("Pitch")
    if yaw_angles:
        data_to_plot.append(yaw_angles)
        labels_box.append("Yaw")
        
    plt.boxplot(data_to_plot, patch_artist=True, labels=labels_box, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    plt.title("Head Pose Angles Variance (Box Plot)", pad=15)
    plt.ylabel("Angle in Degrees")
    plt.tight_layout()
    out_path11 = os.path.join(out_dir, "name_of_graph11.png")
    plt.savefig(out_path11, dpi=300)
    plt.close()

print("All possible non-bar graphs successfully generated and saved into the graphs directory!")
