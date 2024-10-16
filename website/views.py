import os

import numpy as np
from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, current_app
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from .stock_forcasting import StockForecaster
import datetime



views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        note = request.form.get('note')#Gets the note from the HTML

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note
            db.session.add(new_note) #adding the note to the database
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)


@views.route('/forecast', methods=['GET', 'POST'])
@login_required
def forecast_stock():
    if request.method == 'POST':
        ticker = request.form['ticker'].strip()

        # Add validation for ticker
        if not ticker.isalpha():  # Simple validation, more complex checks can be added
            flash('Invalid ticker symbol!', category='error')
            return redirect(url_for('views.forecast_stock'))

        forecaster = StockForecaster(ticker)

        try:
            forecaster.fetch_data()
        except ValueError as e:
            flash(str(e), category='error')
            return redirect(url_for('views.forecast_stock'))

        forecaster.add_technical_indicators()
        df1 = forecaster.prepare_features()
        train_rmse, test_rmse, y_train, y_train_pred, y_test, y_test_pred = forecaster.train_model(df1)
        future_dates, future_forecast = forecaster.forecast_future(steps=30)

        # Save the graphs
        graphs_dir = os.path.join(current_app.root_path, 'static', 'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        # Save the plots to static directory
        static_dir = os.path.join(current_app.root_path, 'static')

        # Closing price history plot
        closing_price_image = os.path.join(graphs_dir, 'closing_price_plot.png')
        forecaster.plot_historical_data(closing_price_image)

        # Train vs Test predictions plot
        train_test_image = os.path.join(graphs_dir, 'train_test_plot.png')
        forecaster.plot_train_vs_test(y_train, y_train_pred, y_test, y_test_pred, train_test_image)

        # Future forecast plot
        future_forecast_image = os.path.join(graphs_dir, 'future_forecast_plot.png')
        forecaster.plot_future_forecast(y_train, y_train_pred, y_test, y_test_pred, future_dates, future_forecast,
                                        future_forecast_image)
        # Render the template with image file paths
        return render_template('forecast.html',
                               ticker=ticker,
                               future_forecast=future_forecast,
                               train_rmse=train_rmse,
                               test_rmse=test_rmse,
                               closing_price_image=url_for('static', filename='closing_price_plot.png'),
                               train_test_image=url_for('static', filename='train_test_plot.png'),
                               future_forecast_image=url_for('static', filename='future_forecast_plot.png'),
                               user=current_user)

    return render_template('forecast.html', user=current_user)



@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})