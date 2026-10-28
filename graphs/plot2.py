import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from datetime import datetime

log_file = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\logs\proctoring.log"

# events_by_date: {date_str: {event_type: count}}
events_by_date = defaultdict(lambda: defaultdict(int))
events_list = set()

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            parts = line.split(" | ", 2)
            if len(parts) == 3:
                date_str = parts[0].split()[0]  # Just the YYYY-MM-DD part
                
                payload = json.loads(parts[2])
                event_type = payload.get("event")
                msg = payload.get("msg") or payload.get("message")
                
                if msg:
                    msg_lower = msg.lower()
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
                    else:
                        continue
                    
                    events_by_date[date_str][evt] += 1
                    events_list.add(evt)
        except Exception as e:
            continue

dates = sorted(events_by_date.keys())
events_list = sorted(list(events_list))

plt.figure(figsize=(10, 6))
plt.style.use("ggplot")

for evt in events_list:
    y_values = [events_by_date[d][evt] for d in dates]
    plt.plot(dates, y_values, marker='o', label=evt, linewidth=2)

plt.title("Trend of Proctoring Violations Over Time", fontsize=16, pad=15)
plt.ylabel("Total Occurrences", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Event Type")
plt.tight_layout()

out_path = r"C:\Users\asama\D\programs\Sem VI\Theory\CV\project_c\graphs\graph2.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_path}")
