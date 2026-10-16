import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained KNN model
MODEL_PATH = "knn.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Attractive HTML layout with Tailwind CSS embedded inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Regressor Prediction Portal</title>
    <!-- Tailwind CSS CDN for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl overflow-hidden p-8">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400">
                KNN Regressor Dashboard
            </h1>
            <p class="text-slate-400 text-sm mt-2">Enter student parameters below to estimate performance metrics.</p>
        </div>

        <!-- Prediction Result Card (Conditional) -->
        {% if prediction is not none %}
        <div class="mb-8 p-5 bg-indigo-950/60 border border-indigo-500/30 rounded-xl text-center animate-pulse">
            <span class="text-xs uppercase tracking-widest text-indigo-300 font-semibold">Predicted Outcome</span>
            <div class="text-3xl font-black text-cyan-400 mt-1">{{ prediction }}</div>
        </div>
        {% endif %}

        <!-- Input Form -->
        <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Age</label>
                <input type="number" step="any" name="age" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Gender (Encoded/Numeric)</label>
                <input type="number" step="any" name="gender" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Course (Encoded/Numeric)</label>
                <input type="number" step="any" name="course" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Study Hours</label>
                <input type="number" step="any" name="study_hours" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Class Attendance</label>
                <input type="number" step="any" name="class_attendance" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Internet Access</label>
                <input type="number" step="any" name="internet_access" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Sleep Hours</label>
                <input type="number" step="any" name="sleep_hours" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Sleep Quality</label>
                <input type="number" step="any" name="sleep_quality" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Study Method</label>
                <input type="number" step="any" name="study_method" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div>
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Facility Rating</label>
                <input type="number" step="any" name="facility_rating" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>
            <div class="md:col-span-2">
                <label class="block text-xs font-medium text-slate-300 uppercase mb-1">Exam Difficulty</label>
                <input type="number" step="any" name="exam_difficulty" required class="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-slate-100 focus:outline-none focus:border-indigo-500 transition">
            </div>

            <!-- Submit Button -->
            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition transform active:scale-[0.99]">
                    Run Prediction
                </button>
            </div>
        </form>

        <div class="mt-8 text-center text-xs text-slate-500">
            Powered by Flask & Scikit-Learn • Deployed on Render
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  if request.method == "POST":
    try:
      # Extract fields matching your model features order
      features = [
          float(request.form["age"]),
          float(request.form["gender"]),
          float(request.form["course"]),
          float(request.form["study_hours"]),
          float(request.form["class_attendance"]),
          float(request.form["internet_access"]),
          float(request.form["sleep_hours"]),
          float(request.form["sleep_quality"]),
          float(request.form["study_method"]),
          float(request.form["facility_rating"]),
          float(request.form["exam_difficulty"]),
      ]

      if model:
        # Reshape to 2D array for prediction
        input_data = np.array([features])
        pred = model.predict(input_data)
        prediction = round(float(pred[0]), 4)
      else:
        prediction = "Model not loaded successfully."
    except Exception as e:
      prediction = f"Error in processing input: {e}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
