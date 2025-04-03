# Introduzione

Il kg agent permette di interagire in modo efficace con un Knowledge Graph. Il suo scopo principale è facilitare la ricerca, l'estrazione e l'organizzazione di informazioni strutturate che descrivono entità e relazioni in un determinato dominio.

## Funzionalità principali

Funzioni essenziali:

1. **Estrazione di informazioni**  
   Consente di recuperare informazioni dettagliate su entità specifiche, come persone, luoghi, oggetti, eventi o concetti astratti, interagendo direttamente con il grafo. Può recuperare attributi, proprietà e collegamenti con altre entità.

2. **Analisi e navigazione delle relazioni**  
   L'agente identifica e analizza le relazioni esistenti tra diverse entità all’interno del grafo. Ciò permette di esplorare connessioni indirette, comprendere pattern ricorrenti e rivelare relazioni nascoste o complesse.

3. **Gestione e aggiornamento del grafo**  
   L'agente consente la modifica del grafo, come l’aggiunta o la modifica di nodi e relazioni. Questa funzione mantiene il grafo aggiornato e adattabile alle esigenze applicative.

## Integrazione tra Database a Grafo e Database Vettoriale

L'agente integra un database a grafo con un database vettoriale, sfruttando i punti di forza di entrambe le tecnologie per migliorare l'elaborazione delle informazioni.

- **Database a Grafo**: Modella e memorizza esplicitamente entità e relazioni, consentendo interrogazioni rapide e intuitive sulle connessioni dirette e indirette.
- **Database Vettoriale**: Rappresenta informazioni non strutturate o semi-strutturate, come testi o immagini, attraverso tecniche di embedding. Ciò permette di effettuare ricerche semantiche basate sulla similarità.

### Processo di Interrogazione Integrata

Quando riceve una richiesta, l'agente può:

- Interrogare il database a grafo per identificare entità e relazioni rilevanti.
- Consultare il database vettoriale per trovare informazioni semanticamente simili.
- Combinare e analizzare i risultati per fornire risposte complete e contestualizzate.

## Architettura e funzionamento operativo

L'agente opera seguendo quattro fasi principali:

1. **Ricezione e interpretazione della richiesta**  
   Traduce le richieste degli utenti in query specifiche eseguibili sui database.

2. **Interrogazione diretta del grafo**  
   Recupera dati specifici, relazioni o pattern di interesse tramite query mirate.

3. **Elaborazione e analisi dei risultati**  
   I dati ottenuti vengono analizzati per identificare significati ulteriori, pattern rilevanti o sintetizzare informazioni complesse.

4. **Restituzione delle informazioni**  
   Le informazioni elaborate vengono restituite chiaramente agli utenti o applicazioni richiedenti, presentandole in un formato facilmente utilizzabile.

# Configurazione 
### Requisiti
- Python (>=3.12)
- Llama Cloud API Key
- OpenAI API Key
- Documenti da indicizzare
- Docker


### Installare le dipendenze con:
```bash
pip install -r requirements.txt
```

### Configurazione neo4j

Per utilizzare il database a grafo, è necessario installare e configurare Neo4j. Puoi farlo seguendo questi passaggi:

eseguire dentro la directory neo4j il comando:
```bash
docker compose up 
```

verra instanziato un docker contsiner di neo4j in locale, per accedere al container puoi andare su [localhost:7474](http://localhost:7474)
al primo accesso inserire le credenziali di default:
```bash
username: neo4j
password: neo4j
```
e successivamente ti verrà chiesto di cambiare la password, inserire come password la seguente:
```bash 
password: llamaindex
```
oppure una password a piacere, ma ricordati di cambiarla anche nel file `settings.py`:

### Preparazione dei dati

#### Caricamento dei documenti
Inserire i documenti da indicizzare nella directory 'data'. [Tipi di documenti supportati](https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types).


### Configurazione dell'environment

creare un file `.env` e inserire le chiavi API necessarie codificate in base64:
```dotenv
OPENAI_API_KEY=
LLAMA_CLOUD_API_KEY=
```

## Utilizzo

### Creazione del grafo e dell'indice vettoriale

Eseguire il file parser.py per creare il grafo l'indice

SE DEVE ESSERE GENERATO UN NUOVO INDICE ELIMINARE PRIMA TUTTO QUELLO CHE SI HA DENTRO LA DIRECTORY
```
storage
```

### Testare l'agente

Eseguire il file kg_agent.py per effettuare il retrieve delle informazioni cambiando prima al fondo del file la domanda a cui vogliamo ottenere una risposta



## Approfondimenti
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Neo4j Documentation]([https://www.trychroma.com/docs/](https://neo4j.com/docs/))
- [Tutorial seguito](https://github.com/run-llama/llama_cloud_services/blob/main/examples/parse/knowledge_graphs/kg_agent.ipynb)
