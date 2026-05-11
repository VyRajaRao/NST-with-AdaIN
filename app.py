import os
from flask import Flask, render_template, request, url_for, send_from_directory, redirect, flash, session
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
from datetime import datetime
from supabase import create_client

import torch
torch.set_num_threads(1)

# Import the existing AdaIN code

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SUPABASE_URL'] = os.getenv('SUPABASE_URL', 'https://vrhukvyqnmhvyofltkqh.supabase.co')
app.config['SUPABASE_ANON_KEY'] = os.getenv(
    'SUPABASE_ANON_KEY',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZyaHVrdnlxbm1odnlvZmx0a3FoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg0NzMzNDgsImV4cCI6MjA5NDA0OTM0OH0.8h3d4QA2QtyGBoyTizpCgThFiRD6WQGcXRSkG1pRIJQ'
)
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

supabase = create_client(app.config['SUPABASE_URL'], app.config['SUPABASE_ANON_KEY'])

class UploadForm(FlaskForm):

    content_image = FileField('Content Image')
    style_image = FileField('Style Image')
    content_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    style_path = HiddenField()
    submit = SubmitField('Transfer Style')


class SignInForm(FlaskForm):
    email = HiddenField('email')
    password = HiddenField('password')

device = torch.device("cpu")

encoder = None
decoder = None

def load_models():
    global encoder, decoder

    if encoder is None or decoder is None:

        encoder = VGGEncoder("vgg_normalised.pth").to(device)

        decoder = Decoder().to(device)
        decoder.load_state_dict(torch.load("experiment/final_exp/decoder_final.pth", map_location=device))
        
        encoder.eval()
        decoder.eval()

    return encoder, decoder

def allowed_file(filename):
    return '.' in filename and \
              filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])
    style_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])

    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_features = encoder(content_image)[-1]
        style_features = encoder(style_image)[-1]

        stylized_features = adaptive_instance_normalization(content_features, style_features)

        stylized_features = alpha * stylized_features + (1 - alpha) * content_features

        stylized_image = decoder(stylized_features)

    return stylized_image

def save_image(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def get_current_user():
    return session.get('user')


def create_user(name, email, password):
    payload = {
        'name': name,
        'email': email.lower().strip(),
        'password_hash': generate_password_hash(password),
        'created_at': datetime.utcnow().isoformat()
    }
    return supabase.table('users').insert(payload).execute()


def find_user_by_email(email):
    return supabase.table('users').select('*').eq('email', email.lower().strip()).limit(1).execute()


def save_history(content_filename, style_filename, result_filename, alpha):
    user = get_current_user()
    if not user:
        return None
    payload = {
        'user_email': user['email'],
        'content_image': content_filename,
        'style_image': style_filename,
        'result_image': result_filename,
        'alpha': float(alpha),
        'created_at': datetime.utcnow().isoformat()
    }
    return supabase.table('gallery_history').insert(payload).execute()


def fetch_history(limit=50):
    user = get_current_user()
    if not user:
        return None
    query = supabase.table('gallery_history').select('*').order('created_at', desc=True).limit(limit)
    query = query.eq('user_email', user['email'])
    return query.execute()


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content_image.data and form.content_image.data.filename:
            if allowed_file(form.content_image.data.filename):
                content_filename = secure_filename(form.content_image.data.filename) 
                form.content_image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style_image.data and form.style_image.data.filename:
            if allowed_file(form.style_image.data.filename):
                style_filename = secure_filename(form.style_image.data.filename) 
                form.style_image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)  
            
            try:
                content_image = Image.open(content_path).convert("RGB")
                style_image = Image.open(style_path).convert("RGB")

                content_image.thumbnail((512, 512))
                style_image.thumbnail((512, 512))

                alpha = float(form.alpha.data)

                encoder, decoder = load_models()
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)


                result_filename = "stylized_" + content_filename.replace(".jpg", "_") + style_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename

                try:
                    history_result = save_history(content_filename, style_filename, result_filename, alpha)
                    if history_result is None:
                        flash('Sign in to save transfers to your gallery history.', 'info')
                except Exception as db_error:
                    flash(f"History save warning: {db_error}", "warning")

            except Exception as e:
                error = str(e)
    elif request.method == "POST":
        if not content_filename:
            error = "Please upload a content image."
        if not style_filename:
            error = "Please upload a style image."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
        user=get_current_user()
    )


@app.route('/gallery')
def gallery():
    history_rows = []
    error = None
    user = get_current_user()
    try:
        result = fetch_history()
        history_rows = result.data or [] if result else []
    except Exception as e:
        error = str(e)
    return render_template('gallery.html', history_rows=history_rows, error=error, user=user)


@app.route('/features')
def features():
    return render_template('features.html', user=get_current_user())


@app.route('/pricing')
def pricing():
    return render_template('pricing.html', user=get_current_user())


@app.route('/contact')
def contact():
    return render_template('contact.html', user=get_current_user())


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        action = request.form.get('action', 'signin')
        name = (request.form.get('name') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return render_template('signin.html', user=get_current_user())

        try:
            if action == 'signup':
                if not name:
                    flash('Name is required for sign up.', 'danger')
                    return render_template('signin.html', user=get_current_user())

                existing = find_user_by_email(email)
                if existing.data:
                    flash('User already exists. Please sign in.', 'warning')
                    return render_template('signin.html', user=get_current_user())

                create_user(name, email, password)
                session['user'] = {'name': name, 'email': email}
                flash('Sign up successful.', 'success')
                return redirect(url_for('index'))

            user_result = find_user_by_email(email)
            user_rows = user_result.data or []
            if not user_rows:
                flash('No account found for this email.', 'danger')
                return render_template('signin.html', user=get_current_user())

            user_row = user_rows[0]
            if not check_password_hash(user_row['password_hash'], password):
                flash('Invalid password.', 'danger')
                return render_template('signin.html', user=get_current_user())

            session['user'] = {'name': user_row.get('name', 'User'), 'email': user_row['email']}
            flash('Signed in successfully.', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Supabase error: {e}', 'danger')

    return render_template('signin.html', user=get_current_user())


@app.route('/signout')
def signout():
    session.pop('user', None)
    flash('Signed out successfully.', 'info')
    return redirect(url_for('index'))


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

