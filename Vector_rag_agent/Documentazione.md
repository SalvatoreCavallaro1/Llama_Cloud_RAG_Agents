# Dynamic Section Retrieval con LlamaIndex

## Introduzione
Questo agente implementa un sistema avanzato di recupero informazioni (Advanced Retrieval-Augmented Generation, RAG) utilizzando LlamaIndex per la selezione dinamica delle sezioni rilevanti di documenti. Il sistema è progettato per migliorare la pertinenza e l'efficacia del recupero di informazioni.


Il **Dynamic Section Retrieval** è una tecnica avanzata di Recupero di Informazioni Aumentato (Retrieval-Augmented Generation, RAG) implementata in **LlamaIndex**, progettata per migliorare la precisione e la coerenza nel recupero di informazioni da documenti strutturati.

Tradizionalmente, i documenti vengono suddivisi in piccoli frammenti (chunk) per facilitare il recupero; tuttavia, questo approccio può interrompere la continuità del contenuto, portando a risposte frammentate o incomplete.

Con il **Dynamic Section Retrieval**, LlamaIndex preserva intere sezioni dei documenti, mantenendo la loro struttura gerarchica e le relazioni contestuali tra le parti. Questo approccio consente di recuperare blocchi di informazioni più coerenti e completi, migliorando la comprensione del contesto e l'accuratezza delle risposte generate.

Il processo si basa su una strategia di recupero in **due fasi**:
1. Vengono identificate le **sezioni rilevanti** del documento.
2. Successivamente, viene recuperato il **contenuto completo** di queste sezioni, garantendo che le informazioni correlate siano mantenute insieme.

Questo metodo riduce il rischio di perdere dettagli importanti che potrebbero essere distribuiti tra diversi frammenti nel tradizionale approccio di chunking.

In sintesi, il **Dynamic Section Retrieval** di LlamaIndex offre un miglioramento significativo nel recupero di informazioni da documenti complessi, assicurando che le risposte siano basate su **contesti completi** e mantenendo l'**integrità delle informazioni originali**.


## Configurazione iniziale

### Requisiti
- Python
- Llama Cloud API Key
- OpenAI API Key
- Microsoft C++ Build Tools
- Documenti da indicizzare


Installare le dipendenze con:
```bash
pip install -r requirements.txt
```

Per utilizzare chroma come database, installare anche: Microsoft C++ Build Tools
Segui questi passaggi per installare Microsoft C++ Build Tools:

1. Vai al sito web di Microsoft Visual C++ Build Tools: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Scarica e installa gli strumenti di build.
3. Durante l'installazione, assicurati di selezionare il componente **"Desktop development with C++"**.


## Preparazione dei dati

### Caricamento dei documenti
Inserire i documenti da indicizzare nella directory 'iclr_docs'. [Tipi di documenti supportati](https://docs.cloud.llamaindex.ai/llamaparse/features/supported_document_types).


## Configurazione dell'environment

creare un file `.env` e inserire le chiavi API necessarie codificate in base64:
```dotenv
OPENAI_API_KEY=
LLAMA_CLOUD_API_KEY=
```

Se si vuole utilizzare Llamatrace per il tracing delle chiamate API, è necessario aggiungere anche la chiave di tracciamento:
```dotenv
OTEL_EXPORTER_OTLP_HEADERS=
PHOENIX_CLIENT_HEADERS=
PHOENIX_COLLECTOR_ENDPOINT=
```

## Utilizzo

### Creazione dell'indice

Eseguire il file parser.py per creare l'indice

SE DEVE ESSERE GENERATO UN NUOVO INDICE ELIMINARE PRIMA TUTTO QUELLO CHE SI HA DENTRO LA DIRECTORY
```
storage_chroma
```

### Testare l'agente

Eseguire il file agent.py per effettuare il retrieve delle informazioni cambiando prima al fondo del file la domanda a cui vogliamo ottenere una risposta


## Risultati
Il sistema recupererà sezioni dinamiche e pertinenti che rispondono in modo specifico alla domanda posta, migliorando significativamente la qualità e l'accuratezza delle risposte.

## Approfondimenti
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Chroma Documentation](https://www.trychroma.com/docs/)
- [Tutorial seguito](https://github.com/run-llama/llama_cloud_services/blob/main/examples/parse/advanced_rag/dynamic_section_retrieval.ipynb)
- [Llamatrace](https://phoenix.arize.com/llamatrace/)**