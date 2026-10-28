"""
Flask Backend — Agamotto
Integrates with existing OpenCV proctoring modules (exam_proctoring_system.py, face_identity.py, etc.)
"""

import os
import sys
import json
import subprocess
import threading
import sqlite3
import base64
import tempfile
import numpy as np
from datetime import datetime, timedelta
from functools import wraps

import bcrypt
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, g
)

# ---------------------------------------------------------------------------
# Path setup — allow importing existing project modules from parent directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'proctor-secret-key-change-in-production')

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

# Tracks running proctoring subprocesses per session { user_id: Popen }
_proctoring_procs = {}
_proctoring_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            password    BLOB NOT NULL,
            face_embedding TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS exam_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            started_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            submitted_at    DATETIME,
            answers         TEXT,
            score           INTEGER DEFAULT 0,
            suspicion_score REAL DEFAULT 0.0,
            status          TEXT DEFAULT 'active',
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS proctoring_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            event_type  TEXT NOT NULL,
            detail      TEXT,
            severity    TEXT DEFAULT 'info',
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES exam_sessions(id)
        );
    """)
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt())


def check_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


# ---------------------------------------------------------------------------
# Questions (hardcoded — easy to swap for DB later)
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": 1,
        "text": "Which data structure uses LIFO (Last In, First Out) ordering?",
        "options": ["Queue", "Stack", "Linked List", "Tree"],
        "answer": 1,
        "topic": "Data Structures"
    },
    {
        "id": 2,
        "text": "What is the time complexity of binary search on a sorted array?",
        "options": ["O(n)", "O(n²)", "O(log n)", "O(1)"],
        "answer": 2,
        "topic": "Algorithms"
    },
    {
        "id": 3,
        "text": "Which of the following is NOT a Python built-in data type?",
        "options": ["list", "dict", "array", "tuple"],
        "answer": 2,
        "topic": "Python"
    },
    {
        "id": 4,
        "text": "In machine learning, overfitting means the model:",
        "options": [
            "Performs well on training and test data",
            "Performs poorly on training data",
            "Performs well on training data but poorly on new data",
            "Cannot converge during training"
        ],
        "answer": 2,
        "topic": "Machine Learning"
    },
    {
        "id": 5,
        "text": "Which HTTP method is idempotent AND safe?",
        "options": ["POST", "PUT", "DELETE", "GET"],
        "answer": 3,
        "topic": "Web"
    },
    {
        "id": 6,
        "text": "What does SQL JOIN without a qualifier default to?",
        "options": ["LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "CROSS JOIN"],
        "answer": 2,
        "topic": "Databases"
    },
    {
        "id": 7,
        "text": "Which sorting algorithm has the best average-case complexity?",
        "options": ["Bubble Sort", "Insertion Sort", "Merge Sort", "Selection Sort"],
        "answer": 2,
        "topic": "Algorithms"
    },
    {
        "id": 8,
        "text": "In CNNs, what does a pooling layer primarily do?",
        "options": [
            "Adds non-linearity",
            "Reduces spatial dimensions",
            "Normalizes activations",
            "Increases feature maps"
        ],
        "answer": 1,
        "topic": "Deep Learning"
    },
    {
        "id": 9,
        "text": "Which of the following achieves thread-safe singleton in Python?",
        "options": [
            "Global variable",
            "threading.Lock() with double-checked locking",
            "Class variable",
            "Module-level import"
        ],
        "answer": 1,
        "topic": "Python"
    },
    {
        "id": 10,
        "text": "What is the purpose of a foreign key constraint in a relational database?",
        "options": [
            "Speeds up queries",
            "Enforces referential integrity between tables",
            "Encrypts sensitive columns",
            "Creates a unique index"
        ],
        "answer": 1,
        "topic": "Databases"
    },
]

EXAM_DURATION_SECONDS = 30 * 60  # 30 minutes


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')

        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()

        if user and check_password(password, user['password']):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['has_face'] = user['face_embedding'] is not None
            if request.is_json:
                return jsonify({'ok': True, 'has_face': session['has_face']})
            return redirect(url_for('dashboard'))

        if request.is_json:
            return jsonify({'ok': False, 'error': 'Invalid credentials'}), 401
        return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username', '').strip()
        email    = data.get('email', '').strip()
        password = data.get('password', '')

        if not (username and email and password):
            msg = 'All fields are required.'
            if request.is_json:
                return jsonify({'ok': False, 'error': msg}), 400
            return render_template('register.html', error=msg)

        if len(password) < 8:
            msg = 'Password must be at least 8 characters.'
            if request.is_json:
                return jsonify({'ok': False, 'error': msg}), 400
            return render_template('register.html', error=msg)

        hashed = hash_password(password)
        db = get_db()
        try:
            db.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed)
            )
            db.commit()
        except sqlite3.IntegrityError:
            msg = 'Username or email already exists.'
            if request.is_json:
                return jsonify({'ok': False, 'error': msg}), 409
            return render_template('register.html', error=msg)

        if request.is_json:
            return jsonify({'ok': True})
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    _stop_proctoring(session.get('user_id'))
    session.clear()
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# Routes — Dashboard & Face Registration
# ---------------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    recent = db.execute(
        '''SELECT * FROM exam_sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT 5''',
        (session['user_id'],)
    ).fetchall()
    return render_template('dashboard.html', user=user, recent_sessions=recent)


@app.route('/api/register-face', methods=['POST'])
@login_required
def register_face():
    """
    Accepts a base64-encoded JPEG frame from the browser webcam,
    runs face_identity.py logic to extract an embedding, and stores it.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'ok': False, 'error': 'No image provided'}), 400

    # Decode base64 image
    try:
        header, encoded = data['image'].split(',', 1)
        img_bytes = base64.b64decode(encoded)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Image decode error: {e}'}), 400

    # Write to temp file so existing face_identity code can read it
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        embedding = _extract_face_embedding(tmp_path)
    finally:
        os.unlink(tmp_path)

    if embedding is None:
        return jsonify({'ok': False, 'error': 'No face detected. Please try again.'}), 422

    # Persist embedding as JSON list
    emb_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
    db = get_db()
    db.execute('UPDATE users SET face_embedding = ? WHERE id = ?',
               (emb_json, session['user_id']))
    db.commit()
    session['has_face'] = True
    return jsonify({'ok': True, 'message': 'Face registered successfully!'})


def _extract_face_embedding(image_path: str):
    """
    Delegate to existing face_identity module.
    Returns numpy array embedding or None.
    """
    try:
        # Import the existing module
        import face_identity as fi  # noqa: F401

        # Common patterns — adapt to your actual function name/signature
        if hasattr(fi, 'get_face_embedding'):
            return fi.get_face_embedding(image_path)
        elif hasattr(fi, 'extract_embedding'):
            return fi.extract_embedding(image_path)
        elif hasattr(fi, 'FaceIdentity'):
            fi_instance = fi.FaceIdentity()
            return fi_instance.get_embedding(image_path)
        else:
            # Fallback: try to read image with OpenCV and call whatever is available
            import cv2
            img = cv2.imread(image_path)
            for attr in dir(fi):
                fn = getattr(fi, attr)
                if callable(fn) and 'embed' in attr.lower():
                    return fn(img)
    except ImportError:
        app.logger.warning('face_identity module not found — using mock embedding for dev')
        # Dev fallback: return a random unit vector so the UI flow works
        v = np.random.rand(128).astype(np.float32)
        return v / np.linalg.norm(v)
    except Exception as e:
        app.logger.error(f'Face embedding error: {e}')
    return None


# ---------------------------------------------------------------------------
# Routes — Exam
# ---------------------------------------------------------------------------

@app.route('/exam')
@login_required
def exam():
    if not session.get('has_face'):
        return redirect(url_for('dashboard'))
    return render_template('exam.html',
                           questions=QUESTIONS,
                           duration=EXAM_DURATION_SECONDS,
                           username=session['username'])


@app.route('/api/start-exam', methods=['POST'])
@login_required
def start_exam():
    """Create exam session and launch proctoring subprocess."""
    db = get_db()

    # Check if user already has an active session
    existing = db.execute(
        "SELECT id FROM exam_sessions WHERE user_id = ? AND status = 'active'",
        (session['user_id'],)
    ).fetchone()

    if existing:
        session_id = existing['id']
    else:
        cur = db.execute(
            'INSERT INTO exam_sessions (user_id) VALUES (?)',
            (session['user_id'],)
        )
        db.commit()
        session_id = cur.lastrowid

    session['exam_session_id'] = session_id

    # Fetch face embedding for proctoring
    user = db.execute('SELECT face_embedding FROM users WHERE id = ?',
                      (session['user_id'],)).fetchone()
    face_emb = user['face_embedding'] if user else None

    # Start proctoring in background
    threading.Thread(
        target=_start_proctoring,
        args=(session['user_id'], session_id, face_emb),
        daemon=True
    ).start()

    return jsonify({'ok': True, 'session_id': session_id})


@app.route('/api/submit-exam', methods=['POST'])
@login_required
def submit_exam():
    """Score exam, store answers, stop proctoring."""
    data = request.get_json()
    answers = data.get('answers', {})  # { "1": 2, "3": 0, ... }
    session_id = session.get('exam_session_id')

    if not session_id:
        return jsonify({'ok': False, 'error': 'No active session'}), 400

    # Calculate score
    score = 0
    for q in QUESTIONS:
        user_ans = answers.get(str(q['id']), -1)
        if user_ans == q['answer']:
            score += 1

    # Pull suspicion score from proctoring events
    db = get_db()
    events = db.execute(
        "SELECT COUNT(*) as cnt FROM proctoring_events WHERE session_id = ? AND severity = 'high'",
        (session_id,)
    ).fetchone()
    high_severity = events['cnt'] if events else 0
    suspicion_score = min(100.0, high_severity * 10.0)

    db.execute(
        """UPDATE exam_sessions
           SET submitted_at = ?, answers = ?, score = ?, suspicion_score = ?, status = 'completed'
           WHERE id = ?""",
        (datetime.now().isoformat(), json.dumps(answers), score, suspicion_score, session_id)
    )
    db.commit()

    _stop_proctoring(session['user_id'])
    session.pop('exam_session_id', None)

    return jsonify({'ok': True, 'redirect': url_for('result', session_id=session_id)})


@app.route('/api/log-event', methods=['POST'])
@login_required
def log_event():
    """Frontend can push client-side events (tab switch, fullscreen exit, etc.)"""
    data = request.get_json()
    session_id = session.get('exam_session_id')
    if not session_id:
        return jsonify({'ok': False}), 400

    db = get_db()
    db.execute(
        'INSERT INTO proctoring_events (session_id, event_type, detail, severity) VALUES (?,?,?,?)',
        (session_id, data.get('type', 'unknown'), data.get('detail', ''), data.get('severity', 'info'))
    )
    db.commit()
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Routes — Results
# ---------------------------------------------------------------------------

@app.route('/result/<int:session_id>')
@login_required
def result(session_id):
    db = get_db()
    exam_session = db.execute(
        'SELECT * FROM exam_sessions WHERE id = ? AND user_id = ?',
        (session_id, session['user_id'])
    ).fetchone()
    if not exam_session:
        return redirect(url_for('dashboard'))

    events = db.execute(
        'SELECT * FROM proctoring_events WHERE session_id = ? ORDER BY ts ASC',
        (session_id,)
    ).fetchall()

    # Reconstruct answered questions for display
    answers = json.loads(exam_session['answers'] or '{}')
    question_results = []
    for q in QUESTIONS:
        user_ans = answers.get(str(q['id']), -1)
        question_results.append({
            'text': q['text'],
            'options': q['options'],
            'correct': q['answer'],
            'user': user_ans,
            'ok': user_ans == q['answer'],
            'topic': q['topic']
        })

    return render_template('result.html',
                           exam_session=exam_session,
                           events=events,
                           question_results=question_results,
                           total=len(QUESTIONS))


# ---------------------------------------------------------------------------
# Proctoring subprocess management
# ---------------------------------------------------------------------------

def _start_proctoring(user_id: int, session_id: int, face_embedding_json: str | None):
    """
    Launch exam_proctoring_system.py as a subprocess, passing user context.
    The subprocess writes events to a shared log; a watcher thread reads them
    and stores them in the DB.
    """
    _stop_proctoring(user_id)  # Kill any stale process

    proctoring_script = os.path.join(PROJECT_ROOT, 'exam_proctoring_system.py')
    if not os.path.exists(proctoring_script):
        app.logger.warning('exam_proctoring_system.py not found — proctoring skipped in dev mode')
        return

    # Write embedding to temp file so subprocess can read it
    emb_file = None
    if face_embedding_json:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, prefix=f'emb_{user_id}_'
        ) as f:
            f.write(face_embedding_json)
            emb_file = f.name

    cmd = [
        sys.executable, proctoring_script,
        '--user-id', str(user_id),
        '--session-id', str(session_id),
        '--db-path', DB_PATH,
    ]
    if emb_file:
        cmd += ['--embedding-file', emb_file]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT
        )
        with _proctoring_lock:
            _proctoring_procs[user_id] = proc

        app.logger.info(f'Proctoring started for user {user_id}, PID {proc.pid}')

        # Watcher: parse stdout events and write to DB
        _watch_proctoring_output(proc, session_id, emb_file)

    except Exception as e:
        app.logger.error(f'Failed to start proctoring: {e}')
        if emb_file and os.path.exists(emb_file):
            os.unlink(emb_file)


def _watch_proctoring_output(proc, session_id: int, emb_file: str | None):
    """
    Read JSON-lines from proctoring subprocess stdout.
    Expected format: {"type": "...", "detail": "...", "severity": "info|medium|high"}
    """
    db_conn = sqlite3.connect(DB_PATH)
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                db_conn.execute(
                    'INSERT INTO proctoring_events (session_id, event_type, detail, severity) VALUES (?,?,?,?)',
                    (session_id, evt.get('type', 'event'),
                     evt.get('detail', ''), evt.get('severity', 'info'))
                )
                db_conn.commit()
            except json.JSONDecodeError:
                # Plain text log line — store as generic event
                db_conn.execute(
                    'INSERT INTO proctoring_events (session_id, event_type, detail) VALUES (?,?,?)',
                    (session_id, 'log', line)
                )
                db_conn.commit()
    finally:
        db_conn.close()
        if emb_file and os.path.exists(emb_file):
            os.unlink(emb_file)


def _stop_proctoring(user_id: int):
    with _proctoring_lock:
        proc = _proctoring_procs.pop(user_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        app.logger.info(f'Proctoring stopped for user {user_id}')


# ---------------------------------------------------------------------------
# API — misc
# ---------------------------------------------------------------------------

@app.route('/api/me')
@login_required
def api_me():
    return jsonify({
        'user_id': session['user_id'],
        'username': session['username'],
        'has_face': session.get('has_face', False)
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    init_db()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
