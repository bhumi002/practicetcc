from flask import Flask, request, jsonify
import pandas as pd
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- File Paths ---
KNOWLEDGE_FILE = "ai_knowledge.json"
main_df = pd.read_excel("Updated_Enhanced_QA_Dataset.xlsx")
ai_df = pd.read_excel("chatbot_help_support_100_questions.xlsx")

# Load extended knowledge
if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "r") as f:
        extra_knowledge = json.load(f)
        extra_df = pd.DataFrame(extra_knowledge)
        ai_df = pd.concat([ai_df, extra_df], ignore_index=True)

# TF-IDF setup
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ai_df["Question"].astype(str))

# --- Organize QA by category ---
qa_data = {}
for _, row in main_df.iterrows():
    category = row.get("Category", "General")
    question = row.get("Question", "")
    answer = row.get("Answer", "")
    qa_data.setdefault(category, []).append({"question": question, "answer": answer})

# Debugging: print the available categories to confirm it's working
print("Available Categories:", list(qa_data.keys()))

def ai_search(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, X).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] > 0.2:
        return ai_df["Answer"].iloc[best_match_index]
    return "I'm not sure. Could you clarify your question?"

def ai_chatbot(user_input):
    lower = user_input.lower()
    if any(greet in lower for greet in ["hello", "hi", "hey"]):
        return "Hello! How can I help you?"
    if "thank" in lower:
        return "You're welcome!"
    return ai_search(user_input)

def update_knowledge(user_q, ai_a):
    global ai_df, X
    new_entry = {"Question": user_q, "Answer": ai_a}
    ai_df = pd.concat([ai_df, pd.DataFrame([new_entry])], ignore_index=True)
    X = vectorizer.fit_transform(ai_df["Question"].astype(str))
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(ai_df.to_dict(orient="records"), f)

# --- API Routes ---
@app.route("/")
def index():
    return "🤖 AI Chatbot API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Missing message"}), 400

    ai_response = ai_chatbot(user_input)
    update_knowledge(user_input, ai_response)

    return jsonify({
        "user": user_input,
        "ai": ai_response
    })

@app.route("/categories", methods=["GET"])
def get_categories():
    # Debugging to check the contents of qa_data
    print("Available Categories:", list(qa_data.keys()))  # Log categories to console
    return jsonify(list(qa_data.keys()))

@app.route("/questions", methods=["GET"])
def get_questions():
    category = request.args.get("category")
    if not category or category not in qa_data:
        print(f"Invalid or missing category: {category}")  # Debug log for category issue
        return jsonify({"error": "Invalid or missing category"}), 400
    return jsonify(qa_data[category])

@app.route("/answer", methods=["POST"])
def get_answer():
    data = request.json
    question = data.get("question", "")
    for cat_questions in qa_data.values():
        for qa in cat_questions:
            if qa["question"] == question:
                return jsonify({"question": question, "answer": qa["answer"]})
    return jsonify({"error": "Question not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
