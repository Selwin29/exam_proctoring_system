"""
exam_proctoring_integration_patch.py
=====================================
This file shows EXACTLY how to modify your existing exam_proctoring_system.py
to work with the Flask backend. Apply only the marked changes — do not rewrite
any of your existing detection logic.

PATCH INSTRUCTIONS:
1. Add argparse CLI args at the top of main()
2. Add JSON event emitter (replaces/wraps your existing logger calls)
3. Pass user_id + face_embedding into FaceIdentity init

──────────────────────────────────────────────────────────────────
STEP 1 — Add these imports at the TOP of exam_proctoring_system.py
──────────────────────────────────────────────────────────────────
"""

# ▼▼▼ PASTE THESE IMPORTS at the TOP of exam_proctoring_system.py ▼▼▼
PATCH_IMPORTS = """
import argparse
import json
import sys
import numpy as np
"""

# ──────────────────────────────────────────────────────────────────
# STEP 2 — Add the argument parser at the START of your main() function
# ──────────────────────────────────────────────────────────────────

PATCH_ARGPARSE = """
def main():
    # ── FLASK INTEGRATION: Parse CLI arguments ──────────────────
    parser = argparse.ArgumentParser(description='AI Agamotto')
    parser.add_argument('--user-id',        type=int,   default=0)
    parser.add_argument('--session-id',     type=int,   default=0)
    parser.add_argument('--db-path',        type=str,   default='')
    parser.add_argument('--embedding-file', type=str,   default='')
    args = parser.parse_args()

    # Load face embedding if provided
    reference_embedding = None
    if args.embedding_file:
        try:
            with open(args.embedding_file, 'r') as f:
                data = json.load(f)
            reference_embedding = np.array(data, dtype=np.float32)
        except Exception as e:
            emit_event('embedding_load_error', str(e), 'medium')

    # ── YOUR EXISTING CODE CONTINUES BELOW (unchanged) ──────────
    # ... your existing init code ...
"""

# ──────────────────────────────────────────────────────────────────
# STEP 3 — Add this emit_event() function (put it near your logger calls)
# It outputs JSON lines to stdout so Flask can read + store them.
# ──────────────────────────────────────────────────────────────────

PATCH_EMIT_FUNCTION = """
def emit_event(event_type: str, detail: str = '', severity: str = 'info'):
    \"\"\"
    Emit a structured event to stdout.
    Flask backend reads these lines and stores them in proctoring_events table.
    Severity: 'info' | 'medium' | 'high'
    \"\"\"
    evt = {
        'type':     event_type,
        'detail':   detail,
        'severity': severity
    }
    print(json.dumps(evt), flush=True)
"""

# ──────────────────────────────────────────────────────────────────
# STEP 4 — Replace / wrap your existing logger calls with emit_event()
#
# Examples:
#   logger.log("Looking away") → emit_event('gaze_away', 'Looking away', 'medium')
#   logger.log("Phone detected") → emit_event('phone_detected', 'Mobile phone in frame', 'high')
#   logger.log("Face not found") → emit_event('face_missing', 'Face not detected', 'high')
#   logger.log("Head pose: down") → emit_event('head_pose', 'Head tilted down', 'medium')
#   logger.log("Multiple faces") → emit_event('multiple_faces', 'More than one face detected', 'high')
#   logger.log("Identity mismatch") → emit_event('identity_fail', 'Face does not match registered user', 'high')
#
# You can KEEP your existing logger calls too — just add emit_event() alongside:
#   logger.log("Phone detected")
#   emit_event('phone_detected', 'Mobile phone visible in frame', 'high')
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# STEP 5 — Pass reference_embedding to FaceIdentity
#
# In your existing init (wherever you create FaceIdentity):
#   face_id = FaceIdentity()  ← existing code
#
# Change to:
#   face_id = FaceIdentity()
#   if reference_embedding is not None:
#       face_id.set_reference_embedding(reference_embedding)
#       # OR: face_id.reference_embedding = reference_embedding
#       # Use whatever setter/attribute your face_identity.py exposes
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# STEP 6 (Optional) — Graceful shutdown on SIGTERM from Flask
# ──────────────────────────────────────────────────────────────────

PATCH_SIGNAL = """
import signal

def _shutdown(sig, frame):
    emit_event('proctor_stopped', 'Proctoring session ended gracefully', 'info')
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
"""

# ──────────────────────────────────────────────────────────────────
# SUMMARY OF CHANGES TO exam_proctoring_system.py
# ──────────────────────────────────────────────────────────────────
SUMMARY = """
MINIMAL CHANGES REQUIRED:
=========================
1. Add:  import argparse, json, sys, numpy as np
2. Add:  argparse block at start of main()
3. Add:  emit_event() function
4. Add:  signal handler for clean shutdown
5. Wrap: each logger.log() call with a matching emit_event() call
6. Pass: reference_embedding → FaceIdentity instance

That's it. No detection logic changed.
"""

print(SUMMARY)
