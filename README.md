# Sentimentanalys av IMDb-filmrecensioner

## Problem och syfte

Det finns enorma mängder recensioner, kommentarer och annan text på nätet, för mycket för att läsa manuellt. Sentimentanalys gör det möjligt att automatiskt avgöra om en text är positiv eller negativ.

Projektet klassificerar filmrecensioner från IMDb som **positiva** eller **negativa** och jämför två modeller:

1. **Logistisk regression med TF-IDF**: en klassisk, snabb baslinje.
2. **DistilBERT** (`distilbert-base-uncased`): en förtränad transformermodell som finjusteras för sentimentklassificering med Hugging Face `transformers`.

Syftet är att undersöka hur mycket en förtränad språkmodell förbättrar resultatet jämfört med en enkel linjär modell, och vad förbättringen kostar i beräkningsresurser.

## Målgrupp

- **Studenter och andra som lär sig NLP**, som vill se en jämförelse mellan en klassisk ML-modell och en transformermodell på samma uppgift.
- **Utvecklare och analytiker** som ska välja en modell för textklassificering och behöver väga träffsäkerhet mot beräkningskostnad.
- **Organisationer med mycket kundtext**, till exempel streamingtjänster, butiker eller recensionssajter, som vill sammanfatta vad användare tycker utan att läsa allt manuellt.

## Resultat

Resultat på testdata:

| Modell                       | Träningsdata | Accuracy | Precision | Recall | F1     |
|------------------------------|--------------|----------|-----------|--------|--------|
| Logistisk regression (TF-IDF)| 20 000       | 0,8809   | 0,8797    | 0,8833 | 0,8815 |
| DistilBERT                   | 5 000        | **0,8971** | **0,8955** | **0,8996** | **0,8976** |

- DistilBERT var bättre på alla mått, men skillnaden var liten (cirka 1,6 procentenheter i accuracy).
- DistilBERT tränades på en fjärdedel så mycket data som logistisk regression. Vi begränsade datan eftersom finjusteringen på CPU tog cirka **2,4 timmar** (en epok, 5 000 recensioner). Logistisk regression tränas på några sekunder.
- Logistisk regression är alltså en stark baslinje med låg beräkningskostnad. DistilBERT ger bäst resultat, men förbättringen måste vägas mot det betydligt högre resursbehovet.

Mer om resultaten, med förväxlingsmatriser och exempel på felklassificerade recensioner, finns i `report.ipynb`.

## Dataset

- **Källa:** [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (Maas et al., 2011), hämtat från Hugging Face som `stanfordnlp/imdb`
- **Storlek:** 50 000 recensioner, varav 25 000 för träning och 25 000 för test
- **Klasser:** 0 = negativ, 1 = positiv (balanserade, 12 500 av varje i träningsdatan)

Datasetet laddas ned automatiskt första gången koden körs, så det behöver inte läggas in manuellt.

### Datauppdelning
- Recensioner som finns i **både** tränings- och testmängden tas bort från testmängden för att undvika dataläckage.
- Träningsmängden delas i 80 % träning (20 000) och 20 % validering (5 000), med `random_state/seed = 42`.
- För DistilBERT görs ett stratifierat urval på 5 000 recensioner ur träningsmängden.

## Projektstruktur

```
projekt_ds/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Laddar datan, tar bort dubbletter, rensar text, delar upp i train/val/test
│   ├── log_reg.py            # TF-IDF-vektorisering och träning av logistisk regression
│   ├── distilbert_model.py   # Tokenisering, modell och Trainer-konfiguration för DistilBERT
│   └── evaluation.py         # Mått, förväxlingsmatris, klassificeringsrapport, felanalys
├── report.ipynb              # Huvudnotebook: EDA, båda modellerna och jämförelse
├── distilbert_veras.ipynb    # Tidigt experiment med DistilBERT (koden flyttades sedan till src/)
├── tmp_trainer/              # Utdata från Hugging Face Trainer (skapas vid körning)
├── requirements.txt
└── .gitignore
```

## Installation

Utvecklat med Python 3.13.

```bash
git clone <repo-URL>
cd projekt_ds

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Användning

Öppna `report.ipynb` och kör cellerna uppifrån och ned. Notebooken:

1. **EDA:** klassfördelning, recensionslängd, recensioner med uttryckligt betyg (t.ex. "8/10") och överlapp mellan tränings- och testdata
2. **Logistisk regression:** träning, utvärdering på validerings- och testdata, felanalys
3. **DistilBERT:** finjustering, utvärdering på testdata, felanalys
4. **Jämförelse** av modellerna

> **Obs:** Finjusteringen av DistilBERT tog cirka 2,4 timmar på CPU. Med GPU går den mycket snabbare. `fp16=True` har bara effekt med GPU.

## Metod

### Förbehandling
- **Logistisk regression:** HTML-taggar (t.ex. `<br />`) tas bort och texten görs om till gemener.
- **DistilBERT:** råtexten skickas direkt till modellens tokenizer (`distilbert-base-uncased`, som själv gör om till gemener). Texten trunkeras till max 512 tokens, och dynamisk padding görs med `DataCollatorWithPadding`.

### Logistisk regression
- `TfidfVectorizer` med standardinställningar, anpassad endast på träningsdatan
- `LogisticRegression(max_iter=1000, random_state=42)`

### DistilBERT
| Parameter        | Värde                     |
|------------------|---------------------------|
| Förtränad modell | `distilbert-base-uncased` |
| Max längd        | 512 tokens                |
| Träningsdata     | 5 000 recensioner         |
| Epoker           | 1                         |
| Batchstorlek     | 32 (träning), 16 (eval)   |
| Inlärningstakt   | 2e-5                      |
| Weight decay     | 0,01                      |

### Utvärdering
Modellerna utvärderas med accuracy, precision, recall, F1-score, förväxlingsmatris och klassificeringsrapport (`src/evaluation.py`). Vi tittar också på de felklassificerade recensioner där modellen var mest säker, för att förstå var den gör fel.

## Begränsningar och vidare arbete

- **Olika mängd träningsdata:** DistilBERT tränades på 5 000 recensioner och logistisk regression på 20 000. Jämförelsen är därför inte helt rättvis, och DistilBERT:s fulla potential undersöktes inte.
- **Ingen hyperparameteroptimering:** båda modellerna kör med standardvärden eller vanliga värden. Logistisk regression skulle kunna förbättras med t.ex. n-gram och justering av `C`.
- **Bara en epok** för DistilBERT, på grund av begränsade beräkningsresurser.
- **Trunkering:** recensioner längre än 512 tokens kortas av, så information i slutet av långa recensioner kan gå förlorad.
- **Vidare arbete:** träna DistilBERT på hela träningsmängden med GPU, testa fler epoker och större modeller (BERT, RoBERTa).

## Gruppmedlemmar

- Carina Karlsson
- Nadine Knoop
- Rebecca Andersson
- Vera Stopp

## Användning av generativ AI

- **Kod och analys:** ChatGPT, Gemini och Claude användes genomgående i projektet. De använts för att diskutera tillvägagångsätt och förenkla dokumentation kring Hugging face Trainer och DistilBERT. Även använts för att göra utkast till DistilBERT-kod. Claude gick igenom skriven README-fil för att se om något hade missats eller behövde ändras. 

## Licens och användningsvillkor

- **Kod:** Projektet har ingen egen licensfil och gjordes som ett skolprojekt.
- **Data:** IMDb-datasetet (Maas et al., 2011) är avsett för forskning och utbildning. Den som använder datasetet ombeds citera:
  > Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng och Christopher Potts. 2011. *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- **Modell:** `distilbert-base-uncased` är licensierad under Apache 2.0.
