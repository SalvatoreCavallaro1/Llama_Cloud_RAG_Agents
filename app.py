import streamlit as st
from Knowledge_graph_agent.kg_agent import agent
################### Streamlit Interface ######################

def main():
    st.title("Agent per Content Driver")
    st.markdown(
        """
        Inserisci la tua domanda riguardo il Content Driver.
        
        """
    )

    # Campo di input per l'utente
    user_input = st.text_input("Inserisci la tua domanda:")

    # Quando si clicca il pulsante "Invia", processa la domanda con l'agente
    if st.button("Invia"):
        if user_input.strip() != "":
            with st.spinner("Elaborazione in corso..."):
                # Chiamata al metodo chat del tuo agente
                response = agent.chat(user_input).response
            st.success("Risposta ottenuta:")
            st.write(response)
        else:
            st.warning("Per favore, inserisci una domanda valida.")

if __name__ == "__main__":
    main()