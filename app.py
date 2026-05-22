"""
SafeShare — Toxic-Free Social Platform
Flask backend with CNN toxicity detection using trained model
"""
import os, json, re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

# Import your toxicity detector
from toxicity_model import get_detector

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-production-please')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///safeshare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Toxicity detection parameters (from your CNN model)
TOXIC_THRESHOLD = 0.552   # from your CNN paper
BAN_THRESHOLD = 10        # violations before account ban

# Initialize toxicity detector (loads your CNN model)
toxicity_detector = get_detector()

# ─────────────────────────────────────────────
# Database models
# ─────────────────────────────────────────────

class User(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    toxic_count  = db.Column(db.Integer, default=0)
    is_banned    = db.Column(db.Boolean, default=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    posts         = db.relationship('Post',         backref='author_user', lazy=True, cascade='all, delete-orphan')
    comments      = db.relationship('Comment',      backref='author_user', lazy=True, cascade='all, delete-orphan')
    notifications = db.relationship('Notification', backref='user',        lazy=True, cascade='all, delete-orphan', order_by='Notification.created_at.desc()')

    def to_dict(self):
        return {
            'id': self.id, 'username': self.username,
            'display_name': self.display_name,
            'toxic_count': self.toxic_count,
            'is_banned': self.is_banned,
            'unread_notifications': sum(1 for n in self.notifications if not n.is_read)
        }


class Post(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content    = db.Column(db.Text, default='')
    image_url  = db.Column(db.String(500), default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    comments = db.relationship('Comment', backref='post', lazy=True, cascade='all, delete-orphan', order_by='Comment.created_at')

    def to_dict(self, current_user_id=None):
        return {
            'id': self.id,
            'author_username': self.author_user.username,
            'author_display': self.author_user.display_name,
            'content': self.content,
            'image_url': self.image_url,
            'created_at': self.created_at.isoformat(),
            'comments': [c.to_dict() for c in self.comments]
        }


class Comment(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    post_id    = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'author_username': self.author_user.username,
            'author_display': self.author_user.display_name,
            'content': self.content,
            'created_at': self.created_at.isoformat()
        }


class Notification(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type       = db.Column(db.String(50), nullable=False)   # toxic_own | toxic_on_post | banned
    message    = db.Column(db.Text, nullable=False)
    is_read    = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id, 'type': self.type,
            'message': self.message, 'is_read': self.is_read,
            'created_at': self.created_at.isoformat()
        }


# ─────────────────────────────────────────────
# Toxicity Analysis (using your trained CNN model)
# ─────────────────────────────────────────────

def analyze_toxicity(text: str) -> dict:
    """
    Use your trained CNN model to detect toxicity
    Returns {'toxic': bool, 'score': float, 'reason': str}
    """
    if toxicity_detector is None or toxicity_detector.model is None:
        # Fallback if model not loaded
        print("[Warning] Toxicity detector not available")
        return {'toxic': False, 'score': 0.0, 'reason': 'Model not loaded'}
    
    try:
        result = toxicity_detector.predict(text)
        return {
            'toxic': result['toxic'],
            'score': result['score'],
            'reason': f'CNN prediction (confidence: {result["confidence"]:.2%})'
        }
    except Exception as e:
        print(f'[Toxicity detection error] {e}')
        return {'toxic': False, 'score': 0.0, 'reason': 'Analysis error'}


# ─────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────

def current_user():
    uid = session.get('user_id')
    return User.query.get(uid) if uid else None

def login_required_json(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user():
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────

@app.route('/')
def index():
    if current_user():
        return redirect(url_for('feed'))
    return render_template('auth.html')

@app.route('/feed')
def feed():
    if not current_user():
        return redirect(url_for('index'))
    return render_template('app.html')


# ─────────────────────────────────────────────
# Routes — Auth API
# ─────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username     = data.get('username', '').strip().lower()
    display_name = data.get('display_name', '').strip()
    password     = data.get('password', '')

    if not username or not display_name or not password:
        return jsonify({'error': 'All fields are required.'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters.'}), 400
    if not re.match(r'^[a-z0-9_]+$', username):
        return jsonify({'error': 'Username can only contain letters, numbers, and underscores.'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken.'}), 400

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, display_name=display_name, password=hashed)
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify({'user': user.to_dict()})


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')
    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid username or password.'}), 401
    session['user_id'] = user.id
    return jsonify({'user': user.to_dict()})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user_id', None)
    return jsonify({'ok': True})


@app.route('/api/me')
def api_me():
    u = current_user()
    if not u:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({'user': u.to_dict()})


# ─────────────────────────────────────────────
# Routes — Posts API
# ─────────────────────────────────────────────

@app.route('/api/posts', methods=['GET'])
@login_required_json
def api_get_posts():
    posts = Post.query.order_by(Post.created_at.desc()).limit(50).all()
    u = current_user()
    return jsonify({'posts': [p.to_dict(u.id) for p in posts]})


@app.route('/api/posts', methods=['POST'])
@login_required_json
def api_create_post():
    u = current_user()
    if u.is_banned:
        return jsonify({'error': 'Your account is banned.'}), 403

    data = request.json
    content   = data.get('content', '').strip()
    image_url = data.get('image_url', '').strip()

    if not content and not image_url:
        return jsonify({'error': 'Post must have text or an image URL.'}), 400

    post = Post(user_id=u.id, content=content, image_url=image_url)
    db.session.add(post)
    db.session.commit()
    return jsonify({'post': post.to_dict(u.id)})


# ─────────────────────────────────────────────
# Routes — Comments API (with toxicity check using CNN)
# ─────────────────────────────────────────────

@app.route('/api/posts/<int:post_id>/comments', methods=['POST'])
@login_required_json
def api_create_comment(post_id):
    u = current_user()
    if u.is_banned:
        return jsonify({'error': 'Your account is banned from commenting.'}), 403

    post = Post.query.get_or_404(post_id)
    data = request.json
    text = data.get('content', '').strip()
    if not text:
        return jsonify({'error': 'Comment cannot be empty.'}), 400
    if len(text) > 1000:
        return jsonify({'error': 'Comment too long (max 1000 characters).'}), 400

    # ── Toxicity check using your CNN model ──────
    result = analyze_toxicity(text)

    if result['toxic']:
        # Increment violation counter
        u.toxic_count += 1

        # Notify commenter
        msg_commenter = (
            f'Your comment was removed: "{text[:60]}{"..." if len(text)>60 else ""}". '
            f'Toxicity score: {round(result["score"]*100)}%. '
            f'Reason: {result.get("reason","—")}. '
            f'This is violation #{u.toxic_count}.'
        )
        db.session.add(Notification(user_id=u.id, type='toxic_own', message=msg_commenter))

        # Notify post owner (if different user)
        if post.user_id != u.id:
            msg_owner = (
                f'@{u.username} posted a toxic comment on your post and it was automatically removed. '
                f'(Score: {round(result["score"]*100)}%)'
            )
            db.session.add(Notification(user_id=post.user_id, type='toxic_on_post', message=msg_owner))

        # Check ban threshold
        just_banned = False
        if u.toxic_count >= BAN_THRESHOLD:
            u.is_banned = True
            just_banned = True
            db.session.add(Notification(
                user_id=u.id, type='banned',
                message=f'Your account has been permanently banned after {BAN_THRESHOLD} toxic comment violations.'
            ))

        db.session.commit()
        return jsonify({
            'toxic': True,
            'score': result['score'],
            'reason': result.get('reason', ''),
            'violation_count': u.toxic_count,
            'banned': u.is_banned,
            'just_banned': just_banned
        }), 200

    # ── Safe comment — save it ──────────────
    comment = Comment(post_id=post_id, user_id=u.id, content=text)
    db.session.add(comment)
    db.session.commit()
    return jsonify({'toxic': False, 'comment': comment.to_dict()})


# ─────────────────────────────────────────────
# Routes — Notifications API
# ─────────────────────────────────────────────

@app.route('/api/notifications', methods=['GET'])
@login_required_json
def api_get_notifications():
    u = current_user()
    notifs = Notification.query.filter_by(user_id=u.id).order_by(Notification.created_at.desc()).all()
    return jsonify({'notifications': [n.to_dict() for n in notifs]})


@app.route('/api/notifications/read', methods=['POST'])
@login_required_json
def api_mark_read():
    u = current_user()
    Notification.query.filter_by(user_id=u.id, is_read=False).update({'is_read': True})
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/notifications', methods=['DELETE'])
@login_required_json
def api_clear_notifications():
    u = current_user()
    Notification.query.filter_by(user_id=u.id).delete()
    db.session.commit()
    return jsonify({'ok': True})


# ─────────────────────────────────────────────
# Health check endpoint (for Render)
# ─────────────────────────────────────────────

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': toxicity_detector is not None and toxicity_detector.model is not None})


# ─────────────────────────────────────────────
# Bootstrap & run
# ─────────────────────────────────────────────

with app.app_context():
    db.create_all()
    # Ensure detector is loaded
    if toxicity_detector and toxicity_detector.model:
        print("✓ Toxicity detector ready with CNN model")
        print(f"  - Threshold: {TOXIC_THRESHOLD}")
        print(f"  - Ban threshold: {BAN_THRESHOLD} violations")
    else:
        print("⚠ Warning: Toxicity detector not loaded - check model files in 'saved_models/' directory")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

