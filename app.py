import streamlit as st
from Knowledge_graph_agent.kg_agent import agent

st.title("💬 Chatbot")
st.caption("🚀 Content Driver chatbot powered by LlamaIndex")

# Inizializza la cronologia dei messaggi se non esiste
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Come posso aiutarti?"}]

# Visualizza la cronologia della chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input del messaggio utente tramite st.chat_input
if prompt := st.chat_input("Scrivi il tuo messaggio"):
    # Aggiungi il messaggio dell'utente alla cronologia e visualizzalo
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Richiama il tuo agente per ottenere la risposta
    with st.spinner("Elaborazione in corso..."):
        # Chiamata al tuo agente (assicurati che agent.chat(prompt) restituisca un oggetto con l'attributo "response")
        response = agent.chat(prompt).response

    # Aggiungi la risposta del tuo agente alla cronologia e visualizzala
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
