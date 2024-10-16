from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db  # Importing db from __init__.py
from flask_login import login_user, login_required, logout_user, current_user
from wtforms.validators import DataRequired, Email, EqualTo, Length

auth = Blueprint('auth', __name__)

# Login Route
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if user:
            # Check if the password is correct
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))  # Redirect to home on success
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    # Render login template with the current_user context
    return render_template("login.html", user=current_user)


# Logout Route
@auth.route('/logout')
@login_required
def logout():
    logout_user()  # Logs out the user
    flash('Logged out successfully!', category='success')  # Flash message on logout
    return redirect(url_for('auth.login'))  # Redirect to login page


# Sign-up Route
@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        # Validation logic
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            # Secure password hashing with pbkdf2:sha256
            new_user = User(
                email=email,
                first_name=first_name,
                password=generate_password_hash(password1, method='pbkdf2:sha256')
            )
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Account created successfully!', category='success')
            return redirect(url_for('views.home'))  # Redirect to home after sign-up

    # Render sign-up template with the current_user context
    return render_template("sign_up.html", user=current_user)
