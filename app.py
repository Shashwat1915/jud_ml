from flask import Flask, request, jsonify
from lamini_analyzer import LaMiniAnalyzer

app = Flask(__name__)
analyzer = LaMiniAnalyzer()

@app.get("/")
def home():
    return "Python ML Service Running!", 200

@app.post("/analyze")
def analyze():
    try:
        text = request.json.get("text", "")
        if not text.strip():
            return jsonify({"clauses": [], "error": "No text provided"}), 400

        chunks, embs = analyzer.fit_document(text)
        clauses = analyzer.extract_key_clauses(text, top_k=5)

        return jsonify({
            "clauses": [c[1] for c in clauses]
        })
    except Exception as e:
        return jsonify({"clauses": [], "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
