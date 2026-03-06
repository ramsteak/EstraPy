# Cose da modificare e aggiungere

## Comando DeGlitch

- deglitch pesato per k

## Comando FileIn

- supportare il formato file per avere #E o # E come header
- aggiunta di --rename per rinominare le pagine, con nomenclatura a variabili simile a save

## Miglioramenti generali

- ri aggiungere plot intermedi per ciascun comando, se richiesti
- modificare da np.polyfit a np.polynomial.x.fit per evitare warning di rank degenerate
- sort dei dati in x in importazione file

## Cose da controllare

- I limiti di range non accettano unità di misura
- Se background o altri comandi vanno oltre le fourier sono malmenate dallo step
- Provare minuit
- Aggiungere deglitch
- Le colonne vanno pulite dei punti per le espressioni
- Finire stili nei plot
- Controllare background che prima di bkg deve essere 0
- Deglitch aggiungere single point removal

## Migliorie

- Gestione degli errori più elegante:
  - In CommandArguments, usare un dizionario per mappare i token agli argomenti del comando.
  - Quando viene generato un errore, di cui costruire una classe specifica, includere Token o Tree[Token] corretta in modo da poter fornire messaggi di errore più dettagliati, con informazioni sulla posizione nel file.

- Irrigidimento di CommandArguments:
  - Implementare la costruzione e validazione di argomenti attraverso dataclasses fields, in modo da ridurre la quantità di codice boilerplate e migliorare la leggibilità.

- Implementare una sezione di help, accessibile anche dall'esterno (`estrapy help <comando>`)

- Sistemare TokenFlow usando eccezioni invede di Enum

## Background

- Aggiungere opzione per fare background in E oltre che in k
