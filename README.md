# Financial Report Intelligence

Assistant IA qui interroge les rapports annuels d'entreprises (10-K), répond aux questions financières en **citant ses sources (document + page)** et **compare les chiffres d'une entreprise d'une année à l'autre**.

Projet du module **AI Engineering**. **Binôme :** Anwar & Ghada

---

## État du projet

| Partie | Responsable | État |
|---|---|---|
| Setup, données FinanceBench, sélection des PDF | Anwar | ✅ terminé |
| Ingestion PDF + métadonnées dans Qdrant (hybride) | Anwar | ✅ terminé |
| Recherche hybride + reranking + citations, RAG | Anwar | ✅ terminé |
| 3 outils, agent LangGraph, serveur MCP | Anwar | ✅ terminé |
| Benchmark complet (récupération + génération) | Ghada | ⏳ à faire |
| Fine-tuning de l'embedding + comparaison avant/après | Ghada | ⏳ à faire |
| Interface Streamlit | Ghada | ⏳ à faire |
| Intégration, rapport technique, démo | Ensemble | ⏳ à faire |

---

## Stack technique

| Composant | Outil |
|---|---|
| LLM | Groq `openai/gpt-oss-120b` via `langchain-groq` |
| Agent | LangGraph (graphe agent ⇄ outils, mémoire `MemorySaver`) |
| Outils | FastMCP (serveur MCP) |
| Base vectorielle | Qdrant (Docker), collection `financial_reports`, mode **hybride** |
| Embeddings denses | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse (BM25) | `Qdrant/bm25` (FastEmbed) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Lecture PDF | `PyPDFLoader` (pypdf) |
| Interface | Streamlit, Plotly |

Toutes les constantes (modèles, URL, chemins, entreprises) sont dans **`src/config.py`**.

---

## Installation (Windows / PowerShell)

### 1. Code et environnement

```powershell
git clone https://github.com/<user>/financial-report-intelligence.git
cd financial-report-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Les versions de `fastmcp`, `mcp`, `langgraph`, `qdrant-client` et `langchain-qdrant` sont **figées** dans `requirements.txt`. Ne pas les mettre à jour, ni installer `langchain-mcp-adapters` : ça casse les dépendances MCP.

### 2. Clé API

Créer `.env` à la racine (jamais commité) :

```
GROQ_API_KEY="gsk_..."
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Clé gratuite sur https://console.groq.com/keys

### 3. Qdrant

Ouvrir Docker Desktop, puis :

```powershell
docker compose up -d
```

Dashboard : http://localhost:6333/dashboard

### 4. Données

```powershell
git clone https://github.com/patronus-ai/financebench.git data/financebench
python src/select_pdfs.py
```

→ copie les PDF des 12 entreprises dans `data/pdfs/` (tous leurs 10-K 2015-2022 + les documents cités dans les questions) et crée `eval/financebench_subset.jsonl` (**82 questions**).

**Entreprises :** PepsiCo, Amcor, Johnson & Johnson, 3M, Best Buy, MGM Resorts, Adobe, JPMorgan, Nike, General Mills, Corning, Boeing.

### 5. Indexation dans Qdrant

```powershell
python src/ingesting.py
```

Long (20 à 40 min). Relancer le script saute les PDF déjà indexés (`file_hash`).
Pour tout réindexer : `python -c "from qdrant_client import QdrantClient; QdrantClient(url='http://localhost:6333').delete_collection('financial_reports')"`

> Chacun de nous a son propre Qdrant local : **Ghada doit faire les étapes 4 et 5** sur sa machine.

---

## Structure du projet

```
financial-report-intelligence/
├── app.py                        # interface Streamlit                         (Ghada)
├── docker-compose.yml
├── requirements.txt
├── .env                          # non versionné
├── data/                         # non versionné : financebench/ et pdfs/
├── src/
│   ├── config.py                 # constantes communes + liste COMPANIES
│   ├── select_pdfs.py            # sélection entreprises / PDF / questions
│   ├── ingesting.py              # PDF -> nettoyage -> chunks + métadonnées -> Qdrant hybride
│   ├── test_retrieval.py         # contrôle rapide page de preuve (top 5)
│   ├── retrieval.py              # search() : hybride + filtre + reranking + citations
│   ├── rag.py                    # answer_question() : RAG avec sources (ligne de commande)
│   ├── tools.py                  # search_filings, get_financial_metric, compare_years
│   ├── agent.py                  # agent LangGraph + fonction ask()
│   ├── mcp_server.py             # serveur MCP (FastMCP) exposant les 3 outils
│   ├── test_mcp.py               # test du serveur MCP
│   └── agent_mcp.py              # agent LangGraph qui utilise les outils via MCP
├── eval/
│   ├── financebench_subset.jsonl # 82 questions (vérité terrain)
│   ├── quick_retrieval_benchmark.py
│   └── quick_retrieval_results.csv
├── notebooks/
└── docs/
```

---

## Utilisation

```powershell
python src/retrieval.py        # test recherche (hybride + reranking)
python src/rag.py              # RAG en ligne de commande : "question | entreprise | année"
python src/tools.py            # test des 3 outils sans agent
python src/agent.py            # agent LangGraph (outils locaux)
python src/test_mcp.py         # test du serveur MCP
python src/agent_mcp.py        # agent LangGraph via le serveur MCP
python eval/quick_retrieval_benchmark.py   # dense / sparse / hybride, avec et sans reranking
```

---

## Conventions communes (le « contrat » backend ⇄ évaluation / interface)

### Métadonnées d'un chunk dans Qdrant

| Champ (`metadata.*`) | Exemple | Origine |
|---|---|---|
| `doc_name` | `3M_2018_10K` | nom du PDF = `doc_name` FinanceBench |
| `company` | `3M` | `financebench_document_information.jsonl` |
| `year` | `2018` (int) | `doc_period` |
| `doc_type` | `10k` | `doc_type` |
| `page` | `55` | PyPDFLoader, **commence à 0** |
| `file_hash` | `abc123...` | SHA256 du PDF |
| `source_file` | `data\pdfs\3M_2018_10K.pdf` | chemin du PDF |

Le texte du chunk est dans `page_content` (nettoyé : espaces et sauts de ligne fusionnés). Chunks de 1000 caractères, chevauchement 100.

### Pages

- **`evidence_page_num` de FinanceBench commence aussi à 0** (vérifié : 8/20 trouvées avec décalage 0, contre 2/20 et 3/20 avec -1 / +1). On compare donc directement `metadata.page == evidence_page_num`.
- Affichage utilisateur et citations : **`page + 1`** (= numéro de page du lecteur PDF).

### Format des citations

`"<Entreprise> <Type> <Année>, p. <page+1>"`, ex. `3M 10-K 2018, p. 56` (fonction `format_citation` dans `retrieval.py`).

---

## API du backend (pour l'interface et l'évaluation)

Toutes les fonctions sont dans `src/`. Depuis `app.py` à la racine : `sys.path.append("src")` avant les imports.

### `retrieval.search(...)` : la recherche

```python
from retrieval import search, get_vector_store

docs = search("net sales", company="3M", year=2018, k=5)            # hybride + reranking
docs = search(question, doc_name="3M_2018_10K", use_reranker=False)  # sans reranking
docs = search(question, store=get_vector_store(RetrievalMode.DENSE))  # dense seul

for d in docs:
    d.page_content            # texte du passage
    d.metadata["citation"]    # "3M 10-K 2018, p. 56"
    d.metadata["page"]        # 55 (base 0)
    d.metadata["score"]       # score du reranker
```

Paramètres : `question, company=None, year=None, doc_name=None, k=5, n_candidates=20, use_reranker=True, store=None`.

### `rag.answer_question(...)` : RAG sans agent

```python
from rag import answer_question
result, docs = answer_question("What were net sales in 2018?", company="3M", year=2018)
result.answer    # "$32,765 million"
result.sources   # ["3M 10-K 2018, p. 56"]
```

Utile pour l'**évaluation de la génération** (réponse + passages utilisés = contexte pour la faithfulness).

### `agent.ask(...)` : l'agent complet (pour l'interface)

```python
from agent import ask
answer, tool_calls = ask("How did Nike's revenues evolve between 2020 and 2022?", thread_id="chat_1")
answer        # texte Markdown avec tableau et sources
tool_calls    # [{"name": "compare_years", "args": {...}, "id": ...}]  (uniquement cette question)
```

- `thread_id` = une conversation (mémoire). Un `thread_id` différent par chat dans Streamlit.
- L'agent choisit seul l'outil, l'entreprise et l'année, reformule la question, répond dans la langue de l'utilisateur et dit qu'il ne sait pas hors périmètre.

### `tools.extract_metric(...)` : un chiffre structuré (pour les graphiques)

`compare_years` renvoie du texte. Pour tracer un graphique, appeler directement :

```python
from tools import extract_metric
r = extract_metric("Nike", 2021, "revenues")
r.value       # 44538.0
r.unit        # "USD millions"
r.citation    # "Nike 10-K 2021, p. 59"
```

Exemple : boucle sur les années, puis `plotly.express.bar` / `line`.

---

## Résultats obtenus

### Récupération : 82 questions, recherche dans le bon document (`doc_name`), page de preuve dans le top 5

| Mode | Reranking | page_recall@5 | MRR |
|---|---|---|---|
| dense | non | 0.537 | 0.343 |
| dense | oui | 0.488 | 0.358 |
| sparse (BM25) | non | 0.402 | 0.286 |
| sparse (BM25) | oui | 0.402 | 0.297 |
| hybride | non | 0.463 | 0.352 |
| **hybride** | **oui** | **0.537** | **0.370** |

→ configuration retenue : **hybride + reranking**. Reranker multilingue du cours (`mmarco-mMiniLMv2`) remplacé par `ms-marco-MiniLM-L-6-v2` (anglais), meilleur sur nos tests.
Limite : 82 questions (1 question ≈ 1,2 point), seule la 1re page de preuve est comptée.

### Agent : tests manuels (11/11 corrects, avec sources)

| Question | Réponse | Outil |
|---|---|---|
| 3M net sales 2018 | 32 765 M$ (p. 56) | get_financial_metric |
| Bénéfice net PepsiCo 2022 (FR) | 8 910 M$ (p. 62) | get_financial_metric |
| Adobe revenue 2021 | 15 785 M$ | get_financial_metric |
| Boeing total assets 2020 | 152 136 M$ | get_financial_metric |
| Best Buy cash 2019 | 1 980 M$ | get_financial_metric |
| General Mills capex 2018 | 623 M$ | get_financial_metric |
| Nike revenues 2020→2022 | 37 403 → 44 538 → 46 710 (+19,1 %, +4,9 %) | compare_years |
| Boeing revenues 2018→2021 | 101 127 → 76 559 → 58 158 → 62 286 | compare_years |
| Risques Boeing 2022 | synthèse Risk Factors avec pages | search_filings |
| 3M 2019 / variation (mémoire) | 32 136 M$, -1,9 % | get_financial_metric |
| Tesla 2021 | « pas dans mes données » | aucun |

### Problèmes rencontrés et solutions (pour le rapport)

| Problème | Cause | Solution |
|---|---|---|
| Texte PDF plein d'espaces / sauts de ligne | extraction PyPDF des tableaux | `clean_text()` avant découpage, réindexation |
| Mauvaise colonne lue dans un tableau large (PepsiCo : 3 166 au lieu de 8 910) | tableaux aplatis en une ligne | modèle 120b + prompt « préférer les états consolidés » ; amélioration possible : extraction des tableaux en Markdown (pymupdf4llm) |
| Question vague (« the company ») → mauvais passages | BM25 sans mots-clés utiles | l'agent **reformule** la requête (query rewriting) |
| Erreur Groq `tool_use_failed` / `json_validate_failed` | sortie structurée instable de gpt-oss-20b | `method="json_schema"`, `with_retry(3)`, passage à gpt-oss-120b |
| `langchain-mcp-adapters` incompatible avec `mcp` 2.x | conflit de versions | adaptateur maison MCP → LangChain dans `agent_mcp.py` (client FastMCP) |

---

## Travail restant (Ghada)

### 1. Benchmark complet (`eval/`)

Base : `eval/quick_retrieval_benchmark.py` et le lab `05-Evaluation` du cours.

- **Récupération** : construire les `qrels` au niveau chunk (comme `02-buidling_qrels.py` du cours : chunks de la page de preuve qui partagent un n-gram avec `evidence_text`), puis **Precision@k, Recall@k, MRR** pour k = 1, 3, 5, 10. Tenir compte de toutes les pages de preuve (`evidence` est une liste).
- **Génération** : pour les 82 questions, `answer_question()` (ou `ask()`), puis comparer avec la colonne `answer` :
  - exactitude (LLM-as-judge Groq : « la réponse est-elle équivalente à la référence ? oui / non ») ;
  - faithfulness (réponse fondée sur les passages : RAGAS ou LLM-as-judge) ;
  - taux de citations présentes et correctes.
- Comparer : RAG simple vs agent.

### 2. Fine-tuning de l'embedding

- Paires (question, passage de preuve) depuis `financebench_subset.jsonl` + chunks Qdrant de la page de preuve.
- **Séparer train / test par entreprise ou par question** avant d'entraîner (sinon résultats faussés).
- `sentence-transformers` + `MultipleNegativesRankingLoss` sur `all-MiniLM-L6-v2` (ou `BAAI/bge-small-en-v1.5`).
- Pour comparer avant / après : nouvelle collection Qdrant (ex. `financial_reports_ft`) indexée avec le modèle fine-tuné (`DENSE_MODEL` = chemin local du modèle, `COLLECTION_NAME` modifié dans `config.py`), puis relancer le benchmark sur les questions de test.

### 3. Interface Streamlit (`app.py`)

Base : `labs/99-exercices/invoices_extractor/app.py` du cours (sidebar, chats, `st.session_state`).

- Chat qui appelle `agent.ask(question, thread_id=<id du chat>)`, réponse affichée en Markdown.
- Sous chaque réponse : outils utilisés (`tool_calls`) dans un `st.expander`.
- Sources cliquables : ouvrir / télécharger le PDF (`data/pdfs/<doc_name>.pdf`) et indiquer la page.
- Onglet « Comparaison » : choix entreprise + indicateur + années → `extract_metric()` par année → graphique Plotly.
- Charger l'agent une seule fois (`@st.cache_resource`), car les modèles sont lourds.

### 4. Ensemble

Intégration et tests de bout en bout, rapport technique (architecture, choix, résultats, limites), slides, démo.

---

## Répartition et planning

| Membre | Responsabilités |
|---|---|
| **Anwar** | Ingestion, recherche hybride, citations, outils, agent LangGraph, serveur MCP |
| **Ghada** | Benchmark, fine-tuning de l'embedding, interface Streamlit |
| **Ensemble** | Architecture, intégration, rapport, démo |

| Semaine | Objectif |
|---|---|
| S1 | Setup, ingestion, recherche hybride avec citations ✅ |
| S2 | Outils, agent LangGraph, MCP ✅ ; benchmark, fine-tuning ⏳ |
| S3 | Interface Streamlit, intégration, rapport et démo |

## Git

On travaille directement sur `main` : **toujours `git pull origin main` avant de coder et avant de pousser**. Fichiers partagés (`config.py`, `requirements.txt`, `README.md`) : se prévenir avant de les modifier.
