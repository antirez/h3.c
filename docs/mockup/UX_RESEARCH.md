# UX research — AI video generation platforms (R29 / T116)

**Fonte e data.** 2026-08-27. Analisi prodotta dall'agente senza accesso web
in tempo reale (`web_search`/`web_fetch` richiedono `ollama signin`, non
eseguito): si basa sulla conoscenza delle piattaforme al training. Ogni voce
è etichettata `[osservato]` (pattern stabile e ripetuto in più versioni della
piattaforma) o `[incerto]` (dettaglio che potrebbe essere cambiato). Da
riverificare con fonti live prima di trattarla come verità definitiva.

Le sette piattaforme analizzate: **Runway** (Gen-3/Gen-4), **Sora**
(inclusi storyboard/remix), **Luma Dream Machine**, **Kling**, **Pika**,
**Google Flow** (Veo), **Hailuo** (MiniMax — stesso produttore del modello
H3, quindi il confronto più diretto).

---

## Pattern osservati → proposta per h3.c Studio

Ogni riga è 1:1: pattern osservato, dove, e la decisione concreta per il
mockup v4 (T117). Rubrica M6, criterio 1: nessun miglioramento nel mockup
senza una riga qui sotto.

| # | Pattern | Piattaforme | h3.c Studio oggi | Proposta v4 |
|---|---|---|---|---|
| P1 | Il prompt è una command bar con **chip di aggancio visivo**: le immagini di riferimento si allegano come tessere direttamente sotto/al bordo del campo di testo, non in una sezione separata | Runway `[osservato]`, Sora `[osservato]`, Luma `[osservato]`, Kling `[incerto]`, Hailuo `[osservato]` | Ancore e riferimenti vivono in schede/pannelli (`PhotoSlot`, `References`) lontani dal prompt | **Adotta**: riga di chip sotto il prompt per first/last frame e fino a N riferimenti; ogni chip apre il picker della libreria sul posto. La frase di R28 resta; i chip sono agganci, non campi. |
| P2 | Scelte rapide come **chip segmentati inline** (aspect, durata, modello) accanto al prompt | Sora `[osservato]`, Luma `[osservato]`, Hailuo `[osservato]` | Le scelte sono parole cliccabili in una riga di testo (R28) | **Non adottare la sostituzione**: la riga-frase è la firma approvata in R28 e misurata (305 px). I chip aprirebbero una griglia proprio dove R28 l'ha tolta. Tenere la frase. |
| P3 | **Galleria a griglia con hover-play**: i video partono in muto al passaggio del mouse; azioni (riusa, scarica, cancella) compaiono in overlay | Runway `[osservato]`, Sora `[osservato]`, Luma `[osservato]`, Pika `[osservato]` | Card statiche con poster; il video parte solo aprendo la presa | **Adotta**: hover-play muto con `preload="metadata"`; le tre azioni già esistenti in overlay sulla card. |
| P4 | Stato di generazione con **preview quasi full-bleed** e stato minimo sovrapposto | Runway `[osservato]`, Sora `[incerto]` | Preview in un riquadro con guida perforata | **Adotta con misura**: nel render la preview diventa l'elemento dominante (già lo è concettualmente); guida perforata e tempo restano, più piccoli. Non full-bleed totale: h3.c è uno strumento che scrive, non un feed. |
| P5 | **"Enhance prompt"**: un pulsante riscrive/espande il prompt con un LLM | Luma `[osservato]`, Hailuo `[osservato]`, Kling `[osservato]` | Assente | **Non in v4**: richiede un LLM nel backend, fuori dall'architettura D20.2 (un job = `./h3` one-shot). Candidato futuro. |
| P6 | **Storyboard/scene multiple** con transizioni fra shot | Flow `[osservato]`, Sora `[incerto]` | Un job = un video | **Non in v4**: h3.c genera una ripresa; multi-shot è prodotto, non UI. |
| P7 | Tema **scuro** dominante, vetro/gradienti | Runway `[osservato]`, Kling `[osservato]`, Hailuo `[osservato]` | Chiaro come progetto (R23), scuro già pronto | **Tenere**: la scelta chiara è deliberata (D23.7) e il dark esiste. Nessun cambio. |
| P8 | **Coda/history in sidebar** persistente visibile durante la composizione | Runway `[incerto]`, Kling `[incerto]` | Striscia di monitoraggio compatta (T96) | **Tenere la striscia**: la pagina a una colonna di R28 non ha spazio per una sidebar senza tradirsi; la striscia dà già composizione-durante-render (R24). |
| P9 | Contatore di **costo/crediti** accanto al pulsante genera | Runway `[osservato]`, Sora `[osservato]`, Luma `[osservato]`, Pika `[osservato]` | Stima in minuti per ogni scelta (T92) | **Già nostro, più forte**: il tempo è l'analogo onesto dei crediti su una GPU locale. Tenere; nel mockup la stima totale sta vicino al pulsante, come i crediti dei concorrenti. |
| P10 | **Drag & drop** di file ovunque nella pagina | Runway `[incerto]`, Luma `[incerto]` | Upload via picker file | **Adotta**: drop sulla pagina = upload nella libreria; costo zero di architettura (l'endpoint T70 esiste). |

**Sintesi delle adozioni per il mockup v4**: P1 (chip di aggancio sotto il
prompt), P3 (hover-play in galleria), P4 (preview dominante nel render),
P9 (stima accanto al genera), P10 (drag & drop). Scartati con motivo: P2,
P5, P6, P7, P8.

## Vincoli che il mockup v4 non deve rompere

- M5: nessun flag CLI e nessun gergo del motore in Create; 100% dei flag in
  Expert; primo video senza aprire pannelli; contrasto AA; stime su ogni
  scelta che cambia il tempo.
- R20/R28: le 42 opzioni restano tutte raggiungibili; la composizione è una
  colonna sola, la frase è la firma.
- Backend invariato: tutto ciò che v4 mostra esiste già nelle API (asset,
  stima, preview, coda, delete). P5 è l'unico pattern che richiederebbe
  backend nuovo, ed è escluso da D29.2/D20.2.
