import os
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import math

folder_path = r"c:\Users\asama\D\programs\Sem VI\Theory\CV\project_c"
graphs_dir = os.path.join(folder_path, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

file_sizes = []
lines_of_code = []
mod_times = []
file_types = defaultdict(int)
file_type_sizes = defaultdict(int)
file_names = []

ignore_dirs = {'.git', '__pycache__', 'venv', 'env', 'runs', 'phone_dataset'}

# Gather folder data
for root, dirs, files in os.walk(folder_path):
    # Skip ignored directories
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    
    for f in files:
        file_path = os.path.join(root, f)
        try:
            size = os.path.getsize(file_path)
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            
            ext = os.path.splitext(f)[1].lower()
            if not ext:
                ext = "no_ext"
                
            file_types[ext] += 1
            file_type_sizes[ext] += size
            
            # Count lines of code if it's a text/code file
            loc = 0
            if ext in {'.py', '.txt', '.yaml', '.md', '.json', '.html', '.css', '.js'}:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_obj:
                        loc = sum(1 for line in f_obj)
                except Exception:
                    pass
                    
            if loc > 0:
                lines_of_code.append(loc)
                file_sizes.append(size / 1024) # KB
                file_names.append(f)
                
            mod_times.append(mod_time)
            
        except OSError:
            pass

# Plotting graphs
plt.style.use('ggplot')

# 1. Scatter Plot: File Size vs Lines of Code (Only for text files)
if lines_of_code and file_sizes:
    plt.figure(figsize=(10, 6))
    plt.scatter(lines_of_code, file_sizes, color='blue', alpha=0.6, s=50)
    plt.title("File Size vs Lines of Code (Scatter Plot)", pad=15)
    plt.xlabel("Lines of Code")
    plt.ylabel("File Size (KB)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "folder_size_vs_loc.png"), dpi=300)
    plt.close()

# 2. Bubble Chart: File Size vs Lines of Code (Bubble size based on log10 of size)
if lines_of_code and file_sizes:
    plt.figure(figsize=(10, 6))
    bubble_sizes = [max(5, (math.log10(max(1, s)) + 1) * 20) for s in file_sizes]
    plt.scatter(lines_of_code, file_sizes, color='purple', alpha=0.5, s=bubble_sizes)
    plt.title("File Size vs Lines of Code (Bubble Chart)", pad=15)
    plt.xlabel("Lines of Code")
    plt.ylabel("File Size (KB)")
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "folder_size_vs_loc_bubble.png"), dpi=300)
    plt.close()

# 3. Pie Chart: File Types Distribution (by count)
if file_types:
    labels = list(file_types.keys())
    sizes = list(file_types.values())
    
    # Sort to keep bottom ones aggregated maybe? No let's just plot all.
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=False)
    plt.title("File Types Distribution by Count (Pie Chart)", pad=20)
    plt.axis('equal')
    plt.savefig(os.path.join(graphs_dir, "folder_file_types_pie.png"), dpi=300)
    plt.close()

# 4. Donut Chart: File Types Distribution (by Size)
if file_type_sizes:
    labels = list(file_type_sizes.keys())
    sizes = [s / (1024*1024) for s in file_type_sizes.values()] # MB
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)
    plt.title("File Types Distribution by Size MB (Donut Chart)", pad=20)
    plt.axis('equal')
    plt.savefig(os.path.join(graphs_dir, "folder_file_sizes_donut.png"), dpi=300)
    plt.close()

# 5. Line Chart: Cumulative Files Created/Modified Over Time
if mod_times:
    mod_times.sort()
    # Create a time series of counts
    counts = list(range(1, len(mod_times) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(mod_times, counts, marker='+', linestyle='-', color='crimson')
    plt.title("Cumulative Modified Files Over Time (Line Plot)", pad=15)
    plt.xlabel("Modification Date/Time")
    plt.ylabel("Cumulative Number of Files")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "folder_mod_times_line.png"), dpi=300)
    plt.close()

# 6. Stacked Area Chart (dummy split by top 3 file types vs others over time)
if mod_times and file_types:
    # We will track cumulative counts of py, pt, and others
    top_types = sorted(file_types.keys(), key=lambda k: file_types[k], reverse=True)[:2]
    
    time_points = sorted(list(set([d.date() for d in mod_times])))
    if len(time_points) > 1:
        type_counts = {t: [0]*len(time_points) for t in top_types}
        type_counts['other'] = [0]*len(time_points)
        
        # This requires re-parsing to match dates to extensions, so we do a quick one:
        file_date_exts = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if not ext: ext = "no_ext"
                t = ext if ext in top_types else 'other'
                try:
                    fp = os.path.join(root, f)
                    dt = datetime.datetime.fromtimestamp(os.path.getmtime(fp)).date()
                    file_date_exts.append((dt, t))
                except:
                    pass
                    
        # Accumulate
        current_counts = {t: 0 for t in type_counts.keys()}
        for i, tp in enumerate(time_points):
            for dt, ext in file_date_exts:
                if dt == tp:
                    current_counts[ext] += 1
            for k in type_counts.keys():
                type_counts[k][i] = current_counts[k]
                
        plt.figure(figsize=(10, 6))
        plt.stackplot(time_points, [type_counts[k] for k in type_counts.keys()], labels=list(type_counts.keys()), alpha=0.7)
        plt.title("Cumulative Files by Top Types Over Time (Area Chart)", pad=15)
        plt.xlabel("Date")
        plt.ylabel("File Count")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "folder_types_area.png"), dpi=300)
        plt.close()

print("Folder statistics graphs generated successfully into graphs directory!")
