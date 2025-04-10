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







