"""
streamlit_app.py  —  Agamotto
────────────────────────────────────────────────────────
UI layer built with streamlit-webrtc for true real-time video.

WHY streamlit-webrtc INSTEAD OF st.rerun():
  st.rerun() rebuilds the entire Streamlit page on every frame (~200-500ms
  overhead per cycle). That's why you see 1 second of video then 5 seconds
  frozen — the page teardown/rebuild time dominates.

  streamlit-webrtc runs the camera and frame processing in a dedicated
  media thread COMPLETELY OUTSIDE Streamlit's rerun cycle. The video
  stream uses WebRTC (browser-native), so frames go directly to the
  browser at full speed. The Streamlit UI panels update at their own
  slower pace — no coupling to video framerate.

INSTALL:
  pip install streamlit streamlit-webrtc opencv-python mediapipe ultralytics av

RUN:
  streamlit run streamlit_app.py
"""

import sys
import os
import time
import cv2
import numpy as np
import streamlit as st
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from exam_proctoring_system import (
    Config, CFG,
    FaceState, PoseResult, PhoneResult,
    detect_face, detect_head_pose, detect_phone, update_score,
    TemporalStateManager, SuspicionScorer, FPSTracker,
    EventHandler,
    draw_face_bbox, draw_pose_axes, draw_phone_detections,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agamotto",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#0d0f14;color:#c8cdd8;}
.main,.block-container{background:#0d0f14;padding:0.8rem 1.2rem;max-width:100%;}
.hdr{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.7rem 1.2rem;margin-bottom:.8rem;}
.hdr-title{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:#e6edf3;letter-spacing:.05em;margin:0;}
.hdr-sub{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:#6e7681;margin:0;}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:.9rem 1rem;margin-bottom:.6rem;}
.card-title{font-family:'IBM Plex Mono',monospace;font-size:.6rem;font-weight:600;color:#6e7681;letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid #21262d;padding-bottom:.35rem;margin-bottom:.6rem;}
.score-num{font-family:'IBM Plex Mono',monospace;font-size:3rem;font-weight:600;line-height:1;margin:.1rem 0;}
.c-ok{color:#3fb950;}.c-warn{color:#d29922;}.c-alrt{color:#f85149;}.c-gray{color:#6e7681;}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-family:'IBM Plex Mono',monospace;font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;}
.b-ok{background:#0d2b1a;color:#3fb950;border:1px solid #238636;}
.b-warn{background:#2b1d0d;color:#d29922;border:1px solid #9e6a03;}
.b-alrt{background:#2b0d0d;color:#f85149;border:1px solid #da3633;}
.b-idle{background:#1c2128;color:#6e7681;border:1px solid #30363d;}
.prog-bg{background:#21262d;border-radius:4px;height:7px;margin:.3rem 0;overflow:hidden;}
.prog-fill{height:100%;border-radius:4px;}
.drow{display:flex;justify-content:space-between;align-items:center;padding:.3rem 0;border-bottom:1px solid #21262d;font-family:'IBM Plex Mono',monospace;font-size:.7rem;}
.drow:last-child{border-bottom:none;}
.dlbl{color:#8b949e;}.dval{font-weight:600;}
.pose-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.3rem;}
.pose-cell{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:.35rem;text-align:center;}
.pose-angle{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;margin:0;}
.pose-lbl{font-size:.58rem;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;margin:0;}
.crow{margin-bottom:.45rem;}
.crow-hdr{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#6e7681;margin-bottom:2px;}
.c-trig{color:#f85149 !important;}
.ev{font-family:'IBM Plex Mono',monospace;font-size:.65rem;padding:.35rem .55rem;border-radius:4px;margin-bottom:.25rem;border-left:3px solid transparent;background:#0d1117;}
.ev-alrt{border-left-color:#da3633;color:#f85149;}
.ev-warn{border-left-color:#9e6a03;color:#d29922;}
.ev-info{border-left-color:#1f6feb;color:#58a6ff;}
footer{display:none;}#MainMenu{display:none;}header[data-testid="stHeader"]{display:none;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (written by media thread, read by UI thread)
# ══════════════════════════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self.face_state   = FaceState()
        self.pose         = PoseResult()
        self.phone_result = PhoneResult()
        self.is_alert     = False
        self.is_warning   = False
        self.fps          = 0.0
        self.score        = 0.0
        self.level        = "OK"
        self.stage_ms     = {}
        self.tsm_snapshot = {}
        self.event_count  = 0
        self.ui_events    = deque(maxlen=10)

if "shared" not in st.session_state:
    st.session_state.shared = SharedState()
SHARED: SharedState = st.session_state.shared


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_models():
    import mediapipe as mp

    CFG.proc_width          = 480
    CFG.proc_height         = 270
    CFG.yolo_imgsz          = 320
    CFG.yolo_every_n_frames = 4

    st.session_state.face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=CFG.mp_max_faces,
        refine_landmarks=CFG.mp_refine,
        min_detection_confidence=CFG.mp_det_conf,
        min_tracking_confidence=CFG.mp_track_conf,
    )

    yolo = None
    try:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo = YOLO(CFG.yolo_model)
        dummy = np.zeros((CFG.yolo_imgsz, CFG.yolo_imgsz, 3), dtype=np.uint8)
        yolo.predict(dummy, device=device, verbose=False)
        CFG.yolo_device = device
    except Exception as e:
        yolo = None
    st.session_state.yolo_model = yolo

    focal = float(CFG.proc_width)
    st.session_state.camera_matrix = np.array(
        [[focal, 0, CFG.proc_width / 2],
         [0, focal, CFG.proc_height / 2],
         [0, 0, 1]], dtype=np.float64)
    st.session_state.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    tsm = TemporalStateManager()
    tsm.register("look_away",   CFG.counter_look_away)
    tsm.register("look_down",   CFG.counter_look_down)
    tsm.register("face_absent", CFG.counter_face_absent)
    tsm.register("phone",       CFG.counter_phone)
    st.session_state.tsm            = tsm
    st.session_state.scorer         = SuspicionScorer(CFG)
    st.session_state.fps_tracker    = FPSTracker(30)
    st.session_state.event_handler  = EventHandler(CFG, fps=CFG.clip_fps)
    st.session_state.pose_ema       = [0.0, 0.0, 0.0]
    st.session_state.prev_centroid  = None
    st.session_state.last_face_seen = time.time()
    st.session_state.frame_idx      = 0
    st.session_state.last_phone     = None
    st.session_state.models_loaded  = True
    CFG.ensure_dirs()


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FRAME CALLBACK  (runs in WebRTC media thread — NOT Streamlit thread)
# ══════════════════════════════════════════════════════════════════════════════

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    ss = st.session_state
    if not ss.get("models_loaded", False):
        return frame

    img  = frame.to_ndarray(format="bgr24")
    proc = cv2.resize(img, (CFG.proc_width, CFG.proc_height))
    rgb  = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

    ss.frame_idx = ss.get("frame_idx", 0) + 1
    fi = ss.frame_idx
    ss.fps_tracker.tick()
    SHARED.fps = ss.fps_tracker.fps

    # Face
    t0 = time.perf_counter()
    face_state, ss.prev_centroid, ss.last_face_seen = detect_face(
        ss.face_mesh, rgb, CFG,
        ss.prev_centroid, ss.last_face_seen,
        CFG.proc_height, CFG.proc_width,
    )
    SHARED.stage_ms["face"] = (time.perf_counter() - t0) * 1000

    # Pose
    t0 = time.perf_counter()
    pose, ss.pose_ema = detect_head_pose(
        face_state, CFG,
        ss.camera_matrix, ss.dist_coeffs,
        ss.pose_ema,
        CFG.proc_height, CFG.proc_width,
    )
    SHARED.stage_ms["pose"] = (time.perf_counter() - t0) * 1000

    # Phone (throttled)
    if ss.yolo_model is not None and fi % CFG.yolo_every_n_frames == 0:
        t0 = time.perf_counter()
        ss.last_phone = detect_phone(
            ss.yolo_model, proc, face_state, CFG,
            CFG.proc_height, CFG.proc_width,
        )
        SHARED.stage_ms["yolo"] = (time.perf_counter() - t0) * 1000

    phone_result = ss.last_phone if ss.last_phone is not None else PhoneResult()

    # Temporal + score
    looking_away = pose.looking_left or pose.looking_right
    ss.tsm.update("look_away",   looking_away)
    ss.tsm.update("look_down",   pose.looking_down)
    ss.tsm.update("face_absent", not face_state.detected)
    ss.tsm.update("phone",       phone_result.phone_detected)

    is_alert, is_warning = update_score(
        ss.scorer, ss.tsm, face_state, pose, phone_result, CFG
    )

    if is_alert:
        _handle_alert(face_state, pose, phone_result, ss.scorer, proc)

    ss.event_handler.push_frame(proc)

    # Draw overlays
    display = proc.copy()
    bbox_color = (
        CFG.C_ALERT if not face_state.detected
        else CFG.C_WARN if is_warning
        else CFG.C_OK
    )
    draw_face_bbox(display, face_state, bbox_color, CFG.proc_height, CFG.proc_width)
    draw_pose_axes(display, pose)
    draw_phone_detections(display, phone_result, CFG.C_ALERT)
    _draw_frame_overlay(display, face_state, pose, phone_result, ss.scorer, SHARED.fps)

    # Write to shared
    SHARED.face_state   = face_state
    SHARED.pose         = pose
    SHARED.phone_result = phone_result
    SHARED.is_alert     = is_alert
    SHARED.is_warning   = is_warning
    SHARED.score        = ss.scorer.score
    SHARED.level        = ss.scorer.level
    SHARED.tsm_snapshot = ss.tsm.snapshot()
    SHARED.event_count  = ss.event_handler.count

    out = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(out, format="rgb24")


def _draw_frame_overlay(frame, face_state, pose, phone_result, scorer, fps):
    h, w = frame.shape[:2]
    level = scorer.level
    if level == "ALERT":
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
        cv2.putText(frame, "  MALPRACTICE ALERT",
                    (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    elif level == "WARNING":
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 70, 150), -1)
        cv2.putText(frame, "  SUSPICIOUS BEHAVIOR",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,170), 1, cv2.LINE_AA)

    sc_col = {"OK": CFG.C_OK, "WARNING": CFG.C_WARN, "ALERT": CFG.C_ALERT}.get(level, CFG.C_INFO)
    cv2.putText(frame, f"Score:{scorer.score:.1f} [{level}]  FPS:{fps:.0f}",
                (6, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sc_col, 2, cv2.LINE_AA)

    pills = []
    if pose.looking_left:          pills.append(("< LEFT",  CFG.C_WARN))
    if pose.looking_right:         pills.append(("RIGHT >", CFG.C_WARN))
    if pose.looking_down:          pills.append(("v DOWN",  CFG.C_WARN))
    if not face_state.detected:    pills.append(("NO FACE", CFG.C_ALERT))
    if face_state.multiple_faces:  pills.append(("MULTI",   CFG.C_ALERT))
    if phone_result.phone_detected: pills.append(("PHONE!", CFG.C_ALERT))

    py = 45
    for text, color in pills:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        px = w - tw - 10
        cv2.rectangle(frame, (px-3, py-th-2), (w-4, py+3), color, -1)
        cv2.putText(frame, text, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
        py += th + 8


def _handle_alert(face_state, pose, phone_result, scorer, frame):
    ss = st.session_state
    active = scorer.active_signals
    reasons = []
    if "phone"       in active: reasons.append("PHONE DETECTED")
    if "face_absent" in active: reasons.append(f"FACE ABSENT {face_state.face_absent_seconds:.1f}s")
    if "look_away"   in active:
        d = "LEFT" if pose.looking_left else "RIGHT"
        reasons.append(f"LOOKING {d} ({pose.yaw:+.0f}°)")
    if "look_down"   in active: reasons.append(f"LOOKING DOWN ({pose.pitch:+.0f}°)")
    if "multi_face"  in active: reasons.append(f"MULTIPLE FACES ({face_state.num_faces})")

    reason_str = " | ".join(reasons) or "Sustained suspicious behavior"
    event = ss.event_handler.handle_alert(reason_str, scorer.score, active, frame, CFG)
    ts = time.strftime("%H:%M:%S")
    SHARED.ui_events.appendleft({
        "ts": ts,
        "level": "ALERT" if event else "WARNING",
        "msg": reason_str,
        "score": scorer.score,
    })


# ══════════════════════════════════════════════════════════════════════════════
# UI PANEL RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _badge(level):
    cls = {"OK":"b-ok","WARNING":"b-warn","ALERT":"b-alrt"}.get(level,"b-idle")
    return f'<span class="badge {cls}">{level}</span>'

def _prog(ratio, triggered):
    pct = min(int(ratio*100), 100)
    color = "#f85149" if triggered else "#388bfd"
    return f'<div class="prog-bg"><div class="prog-fill" style="width:{pct}%;background:{color};"></div></div>'

def render_score_panel():
    s = SHARED
    cls = {"OK":"c-ok","WARNING":"c-warn","ALERT":"c-alrt"}.get(s.level,"c-ok")
    bar = {"OK":"#3fb950","WARNING":"#d29922","ALERT":"#f85149"}.get(s.level,"#3fb950")
    pct = min(int((s.score/CFG.alert_threshold)*100), 100)
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Suspicion Score</div>
      <div class="score-num {cls}">{s.score:.1f}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#6e7681;margin-bottom:.3rem;">
        / {CFG.alert_threshold:.0f} &nbsp; warn at {CFG.warning_threshold:.0f}
      </div>
      <div class="prog-bg"><div class="prog-fill" style="width:{pct}%;background:{bar};"></div></div>
      <div style="margin-top:.45rem;">{_badge(s.level)}</div>
    </div>""", unsafe_allow_html=True)

def render_detection_panel():
    s=SHARED; fs=s.face_state; pr=s.phone_result
    if not fs.detected:            fv,fc=f"ABSENT {fs.face_absent_seconds:.1f}s","c-alrt"
    elif fs.multiple_faces:        fv,fc=f"MULTIPLE ({fs.num_faces})","c-warn"
    else:                          fv,fc="DETECTED","c-ok"
    if pr.phone_detected and pr.best: pv,pc=f"DETECTED {pr.best['conf']:.0%}","c-alrt"
    else:                             pv,pc="Not detected","c-ok"
    fc2="c-ok" if s.fps>20 else "c-warn" if s.fps>10 else "c-alrt"
    pose=s.pose
    flags=[]
    if pose.looking_left:  flags.append('<span class="c-warn">◄ LEFT</span>')
    if pose.looking_right: flags.append('<span class="c-warn">RIGHT ►</span>')
    if pose.looking_down:  flags.append('<span class="c-warn">▼ DOWN</span>')
    if pose.tilted:        flags.append('<span class="c-warn">TILTED</span>')
    fh=" &nbsp; ".join(flags) if flags else '<span class="c-gray">—</span>'
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Detection Status</div>
      <div class="drow"><span class="dlbl">FACE</span><span class="dval {fc}">{fv}</span></div>
      <div class="drow"><span class="dlbl">PHONE</span><span class="dval {pc}">{pv}</span></div>
      <div class="drow"><span class="dlbl">GAZE FLAGS</span><span class="dval" style="font-size:.65rem;">{fh}</span></div>
      <div class="drow"><span class="dlbl">FPS</span><span class="dval {fc2}">{s.fps:.1f}</span></div>
      <div class="drow"><span class="dlbl">EVENTS SAVED</span><span class="dval c-gray">{s.event_count}</span></div>
    </div>""", unsafe_allow_html=True)

def render_pose_panel():
    pose=SHARED.pose
    def _f(v): return f"{v:+.1f}°" if pose.valid else "—"
    def _c(f): return "c-warn" if f and pose.valid else "c-gray"
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Head Pose (solvePnP + EMA)</div>
      <div class="pose-grid">
        <div class="pose-cell"><p class="pose-angle {_c(pose.looking_left or pose.looking_right)}">{_f(pose.yaw)}</p><p class="pose-lbl">Yaw</p></div>
        <div class="pose-cell"><p class="pose-angle {_c(pose.looking_down or pose.looking_up)}">{_f(pose.pitch)}</p><p class="pose-lbl">Pitch</p></div>
        <div class="pose-cell"><p class="pose-angle {_c(pose.tilted)}">{_f(pose.roll)}</p><p class="pose-lbl">Roll</p></div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_counter_panel():
    snap=SHARED.tsm_snapshot
    labels={"look_away":"Look Away","look_down":"Look Down","face_absent":"No Face","phone":"Phone"}
    rows=""
    for key,lbl in labels.items():
        d=snap.get(key,{"value":0,"fill":0,"triggered":False})
        trig=d.get("triggered",False)
        lc="c-trig" if trig else ""
        rows+=f'<div class="crow"><div class="crow-hdr"><span class="{lc}">{lbl}</span><span class="{lc}">{d.get("value",0):.0f}</span></div>{_prog(d.get("fill",0),trig)}</div>'
    st.markdown(f'<div class="card"><div class="card-title">Temporal Counters</div>{rows}</div>', unsafe_allow_html=True)

def render_event_log():
    events=list(SHARED.ui_events)
    if not events:
        st.markdown('<div class="card"><div class="card-title">Event Log</div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:.68rem;color:#6e7681;">No events recorded.</div></div>', unsafe_allow_html=True)
        return
    rows=""
    for e in events[:8]:
        cls={"ALERT":"ev-alrt","WARNING":"ev-warn"}.get(e.get("level","INFO"),"ev-info")
        rows+=f'<div class="ev {cls}">[{e["ts"]}] {e["msg"]} <span style="opacity:.55;">score={e["score"]:.1f}</span></div>'
    st.markdown(f'<div class="card"><div class="card-title">Event Log</div>{rows}</div>', unsafe_allow_html=True)

def render_timing():
    ms=SHARED.stage_ms
    if not ms: return
    parts=" &nbsp;|&nbsp; ".join(f'<span style="color:#6e7681">{k}</span> <span style="color:#58a6ff">{v:.0f}ms</span>' for k,v in ms.items())
    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.6rem;color:#6e7681;padding:.2rem 0;">{parts}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div class="hdr">
      <div>
        <p class="hdr-title">🎓 AGAMOTTO</p>
        <p class="hdr-sub">MediaPipe · YOLOv8 · solvePnP · WebRTC  —  real-time, no lag</p>
      </div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.get("models_loaded", False):
        with st.spinner("Loading models — please wait…"):
            _load_models()

    left_col, right_col = st.columns([3, 2], gap="medium")

    with left_col:
        st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#6e7681;margin-bottom:.3rem;letter-spacing:.1em;">LIVE FEED  —  click START to begin</div>', unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="proctoring",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
                "audio": False,
            },
            async_processing=True,
        )

        timing_slot = st.empty()

        if st.button("↺ Reset Score & Counters", use_container_width=True):
            ss = st.session_state
            if ss.get("scorer"): ss.scorer.reset()
            if ss.get("tsm"):    ss.tsm.reset_all()
            SHARED.ui_events.clear()
            ss.pose_ema      = [0.0, 0.0, 0.0]
            ss.prev_centroid = None
            st.success("Reset.")

    with right_col:
        score_slot   = st.empty()
        detect_slot  = st.empty()
        pose_slot    = st.empty()
        counter_slot = st.empty()
        log_slot     = st.empty()

    # ── Live update loop (only runs while stream is active) ───────────────
    # The video runs in a separate thread via WebRTC.
    # This loop only refreshes the right-column stat panels at ~4Hz.
    # No st.rerun() needed — placeholder updates happen in-place.
    if ctx.state.playing:
        while True:
            with timing_slot.container():  render_timing()
            with score_slot.container():   render_score_panel()
            with detect_slot.container():  render_detection_panel()
            with pose_slot.container():    render_pose_panel()
            with counter_slot.container(): render_counter_panel()
            with log_slot.container():     render_event_log()
            time.sleep(0.25)
    else:
        with score_slot.container():   render_score_panel()
        with detect_slot.container():  render_detection_panel()
        with pose_slot.container():    render_pose_panel()
        with counter_slot.container(): render_counter_panel()
        with log_slot.container():     render_event_log()


if __name__ == "__main__":
    main()