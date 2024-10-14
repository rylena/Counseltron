from flask import Flask, render_template, request, jsonify
from counseltron import get_greeting_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request. 'message' key missing."}), 400

    text = data.get("message")
    bot_response = get_greeting_response(text)
    message = {"answer": bot_response}
    
    return jsonify(message)

if __name__ == "__main__":
    # Change host     '0.0.0.0' to allow access from other devices
    app.run(host='0.0.0.0', port=5000, debug=True)
















