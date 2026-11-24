# Financial Report Intelligence

Assistant IA qui interroge les rapports annuels d'entreprises (10-K), répond aux questions financières en **citant ses sources (document + page)** et **compare les chiffres d'une entreprise d'une année à l'autre**.

Projet réalisé dans le cadre du module **AI Engineering**.

**Binôme :** Anwar & Ghada

---

## Fonctionnalités

- **Ingestion PDF** : découpage des rapports en chunks, enrichis de métadonnées (entreprise, année, type de document, page).
- **RAG hybride** : recherche dense (embeddings) + sparse (BM25) dans Qdrant, filtre par entreprise et par année, reranking avec un CrossEncoder.
- **Citations** : chaque réponse indique sa source (ex. *3M, 10-K 2018, p. 52*). Sans preuve, l'assistant répond qu'il ne sait pas.
- **Agent LangGraph + outils MCP** :
  - `search_filings` : recherche de passages dans les rapports ;
  - `get_financial_metric` : extraction d'un indicateur (chiffre d'affaires, résultat net…) pour une année ;
  - `compare_years` : comparaison d'un indicateur sur plusieurs années.
- **Fine-tuning** d'un modèle d'embedding sur des paires question–passage de FinanceBench.
- **Benchmark** : récupération (Precision@k, Recall@k, MRR) et génération (exactitude, fidélité aux sources).
- **Interface web** Streamlit : chat, sources, graphiques de comparaison.

## Stack technique

| Composant | Outil |
|---|---|
| LLM | Groq (`openai/gpt-oss-20b`) via `langchain-groq` |
| Orchestration | LangChain, LangGraph |
| Outils | FastMCP |
| Base vectorielle | Qdrant (Docker) |
| Embeddings denses | `sentence-transformers/all-MiniLM-L6-v2` |
| Embeddings sparse (BM25) | `Qdrant/bm25` (FastEmbed) |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Lecture PDF | `PyPDFLoader` (pypdf) |
| Interface | Streamlit, Plotly |

## Structure du projet

```
financial-report-intelligence/
├── app.py                  # interface Streamlit
├── docker-compose.yml      # Qdrant
├── requirements.txt
├── .env.example
├── data/                   # non versionné (voir « Données »)
│   ├── financebench/
│   └── pdfs/
├── src/
│   ├── config.py           # constantes communes
│   ├── select_pdfs.py      # sélection des entreprises et des PDF
│   ├── ingesting.py        # PDF -> chunks + métadonnées -> Qdrant
│   ├── retrieval.py        # recherche hybride, filtres, reranking
│   ├── tools.py            # search_filings, get_financial_metric, compare_years
│   ├── agent.py            # agent LangGraph
│   └── mcp_server.py       # serveur MCP
├── eval/                   # benchmark et fine-tuning
├── notebooks/              # explorations
└── docs/                   # cahier des charges, schémas
```

## Installation

### 1. Cloner le repo et créer l'environnement

```powershell
git clone https://github.com/<user>/financial-report-intelligence.git
cd financial-report-intelligence
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
```

### 2. Clé API

Copier `.env.example` en `.env` et renseigner la clé Groq :

```
GROQ_API_KEY="your groq API key"
```

### 3. Lancer Qdrant

Ouvrir Docker Desktop, puis :

```bash
docker compose up -d
```

Dashboard : http://localhost:6333/dashboard

## Données

Le projet utilise le benchmark [FinanceBench](https://github.com/patronus-ai/financebench) (questions financières avec réponses et pages de preuve, et les rapports PDF associés).

### 1. Télécharger FinanceBench

À la racine du projet :

```bash
git clone https://github.com/patronus-ai/financebench.git data/financebench
```

### 2. Sélectionner les entreprises et copier les PDF

```bash
python src/select_pdfs.py
```

Ce script :
- garde les questions des entreprises listées dans `COMPANIES` (`src/config.py`) ;
- copie leurs PDF dans `data/pdfs/` ;
- génère `eval/financebench_subset.jsonl` pour l'évaluation.

**Entreprises retenues (12) :** PepsiCo, Amcor, Johnson & Johnson, 3M, Best Buy, MGM Resorts, Adobe, JPMorgan, Nike, General Mills, Corning, Boeing.
Soit **82 questions** et leurs rapports (10-K de 2015 à 2022 pour la plupart).

## Métadonnées d'un chunk (Qdrant)

Format commun entre l'ingestion et l'évaluation / l'interface :

| Champ | Exemple | Origine |
|---|---|---|
| `doc_name` | `3M_2018_10K` | nom du PDF (identique à FinanceBench) |
| `company` | `3M` | `financebench_document_information.jsonl` |
| `year` | `2018` | colonne `doc_period` |
| `doc_type` | `10k` | colonne `doc_type` |
| `page` | `51` | ajouté par `PyPDFLoader` (commence à 0) |
| `file_hash` | `abc123...` | SHA256 du PDF (évite la ré-indexation) |
| `source_file` | `data/pdfs/3M_2018_10K.pdf` | chemin du PDF |

Affichage à l'utilisateur : `page + 1`.

## Utilisation

> Sections complétées au fil du projet.

```bash
python src/ingesting.py          # indexer les PDF dans Qdrant
streamlit run app.py             # lancer l'interface
```

## Évaluation

> À compléter avec les résultats : Precision@k, Recall@k, MRR (dense / sparse / hybride, avec et sans reranking), puis avant / après fine-tuning.

## Répartition des tâches

| Membre | Responsabilités |
|---|---|
| **Anwar** | Ingestion PDF et métadonnées, recherche hybride et citations, outils MCP, agent LangGraph |
| **Ghada** | Préparation de FinanceBench, benchmark, fine-tuning de l'embedding, interface Streamlit |
| **Ensemble** | Architecture, intégration, rapport technique, démo |

## Planning

| Semaine | Objectif |
|---|---|
| S1 | Setup, ingestion, recherche hybride avec citations, métriques de récupération |
| S2 | Outils MCP, agent LangGraph, fine-tuning, évaluation de la génération |
| S3 | Interface Streamlit, intégration, rapport et démo |
