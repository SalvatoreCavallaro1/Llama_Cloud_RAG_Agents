# flask_server.py
from flask import Flask, request, jsonify
from Knowledge_graph_agent.kg_agent import create_agent_instance
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Dizionario per memorizzare le istanze agente in base al session_id
sessions = {}

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint che gestisce la richiesta di una conversazione.
    Il JSON atteso deve avere i campi:
      - "session_id": identificativo della sessione (necessario per gestire conversazioni multiple)
      - "query": la domanda/informazione da inviare all'agente
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Manca il campo 'query' nel JSON."}), 400

    # Recuperiamo sia la query che il session_id
    user_query = data["query"]
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "Manca il campo 'session_id' nel JSON."}), 400

    # Se non esiste già una sessione per questo id, creiamo una nuova istanza dell'agente
    if session_id not in sessions:
        sessions[session_id] = create_agent_instance()

    agent = sessions[session_id]

    try:
        # Utilizziamo il metodo chat per interagire con l'agente
        response_obj = agent.chat(user_query)
        output_text = getattr(response_obj, "response", None)
        if not output_text:
            output_text = str(response_obj)
        return jsonify({"response": output_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Avvia il server Flask sulla porta 5000
    app.run(host='0.0.0.0', port=5000, debug=True)

