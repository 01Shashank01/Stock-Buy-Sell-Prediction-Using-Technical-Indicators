from flask import Flask, render_template, request
from model_pipeline import run_model
import os
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    chart_paths = []

    if request.method == 'POST':
        try:
            stock = request.form['stock']
            interval = request.form['interval']
            window = int(request.form['window'])

            result = run_model(stock, interval, window)

            if result and result.get("success"):
                predictions = result.get("prediction", [])
                chart_paths = result.get("image_paths", [])
            else:
                # Log the error but don't show it on the site
                print("Model returned error:", result.get("error", "Unknown error"))

        except Exception:
            print("Exception occurred in run_model:")
            traceback.print_exc()  # Keep it in logs for debugging

    return render_template('index.html', predictions=predictions, chart_paths=chart_paths)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)










'''
# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from model_pipeline import run_model
import os
import json
import traceback

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'change_this!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

# Listen to event from client-side JavaScript
@socketio.on('run_model_event')
def handle_run_model_event(payload):
    print("Received payload from client:", payload)
    try:
        # 1. Extract user input
        stock = payload['stock']
        interval = payload['interval']
        window = int(payload['window'])
        indicators = payload.get('selected_indicators', [])

        # 2. Call the model pipeline with user inputs
        result = run_model(stock, interval, window, indicators)

        if not result.get("success"):
            raise RuntimeError(result.get("error", "Unknown error"))

        # 3. Extract predictions and chart JSONs
        predictions = result["prediction"]
        chart_jsons = result["chart_jsons"]
        chart_dicts = [json.loads(js) for js in chart_jsons]

        # 4. Split global prediction info and indicator-based predictions
        global_info = predictions[:2]  # e.g., buy/sell recommendation and confidence
        indicator_predictions = predictions[2:]

        grouped_predictions = []
        idx = 0
        for indicator in indicators:
            size = {
                'macd': 2,
                'rsi': 2,
                'adx': 1,
                'bollinger': 2,
                'support_resistance': 4
            }.get(indicator, 2)
            grouped_predictions.append(indicator_predictions[idx:idx+size])
            idx += size

        # 5. Zip charts and their corresponding predictions
        charts_and_predictions = list(zip(chart_dicts, grouped_predictions))

        # 6. Emit result back to the frontend
        emit('model_response', {
            'success': True,
            'global_info': global_info,
            'charts_and_predictions': charts_and_predictions
        })

    except Exception as e:
        traceback.print_exc()
        emit('model_response', {
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    socketio.run(app, debug=True)'''

