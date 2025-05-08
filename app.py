from flask import Flask, request, jsonify, send_from_directory
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize the Ollama LLM (ensure Ollama is running locally or adjust as needed)
llm = Ollama(model="phi:latest")

# System prompt for the AI counselor
SYSTEM_PROMPT = """
You are Counseltron, an empathetic, knowledgeable, and supportive AI student counselor. Help students with academic, career, and personal advice. Always be encouraging and provide actionable suggestions.
"""

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided.'}), 400

    # Compose the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", user_message)
    ])

    # Get response from LLM
    response = llm.invoke(prompt.format())
    # Remove lines starting with 'Think:'
    filtered_response = '\n'.join(line for line in response.splitlines() if not line.strip().lower().startswith('think:'))
    return jsonify({'response': filtered_response.strip()})

@app.route('/')
def serve_frontend():
    return send_from_directory('Counseltron', 'frontend.html')

@app.route('/Counseltron/<path:filename>')
def serve_static(filename):
    return send_from_directory('Counseltron', filename)

if __name__ == '__main__':
    app.run(debug=True)
