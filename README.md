# UdaPlay - AI Game Research Agent

UdaPlay is an AI research agent that answers natural-language questions about video games: titles, release dates, platforms, genres, descriptions and publishers.

It uses a two-tier retrieval system:

1. **Primary - RAG** over a local dataset of 15 video games, stored in a persistent ChromaDB vector database.
2. **Secondary - web search** with the Tavily API, used only when an LLM judge decides the local knowledge is not enough.

Every answer comes with a confidence level and cites its sources (with URLs for web sources). The agent remembers previous questions within a session and, as an optional extension, keeps what it learns from the web in a persistent long-term memory.

## Getting Started

The project has two notebooks, both run from the `starter/` folder:

- `Udaplay_01_starter_project.ipynb` - **Part 1, offline RAG**: indexes the games into ChromaDB and demonstrates semantic search.
- `Udaplay_02_starter_project.ipynb` - **Part 2, agent**: the UdaPlay agent with its tools, state machine and reports.

Part 2 reuses the database created by Part 1, so run Part 1 first.

### Dependencies

Python 3.11 or later, plus the libraries in `starter/requirements.txt`:

```
chromadb>=1.0.4
openai>=1.73.0
pydantic>=2.11.3
python-dotenv>=1.1.0
tavily-python>=0.5.4
pdfplumber>=0.11.6
jupyter
ipykernel
```

You also need:

- an OpenAI API key (in the Udacity workspace, the `voc-` key from the Cloud Resources panel);
- a Tavily API key (free account at https://app.tavily.com/home).

### Installation

1. Create and activate a virtual environment in `starter/`:

   ```
   cd starter
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create the `.env` file from the template and fill in the keys:

   ```
   cp .env.example .env
   ```

   ```
   OPENAI_API_KEY="voc-..."
   OPENAI_BASE_URL="https://openai.vocareum.com/v1"
   TAVILY_API_KEY="tvly-..."
   ```

   The file must be named exactly `.env`. It is ignored by git: never commit it.

4. Start Jupyter from `starter/` and run the notebooks in order:

   ```
   jupyter notebook
   ```

   Part 1 creates the `starter/chromadb/` folder with the `udaplay` collection. Do not delete it, or re-run Part 1 before Part 2.

## Testing

There is no automated test suite: the notebooks are the demonstration. Run every cell of both notebooks in order. Each run calls the OpenAI and Tavily APIs, so answers and web sources can vary between runs.

Part 2 empties the long-term memory at the start (`RESET_LONG_TERM_MEMORY = True`), so every run shows the web fallback first and then the memory hit.

### Break Down Tests

**Part 1 - semantic search.** Five queries check that the vector database returns the right games, plus a reload check:

```
When was Pokémon Gold and Silver released?          -> Pokémon Gold and Silver first
Which one was the first 3D platformer Mario game?   -> Super Mario 64 first
Who published Grand Theft Auto: San Andreas?        -> Grand Theft Auto: San Andreas first
Racing games on PlayStation                         -> the two Gran Turismo games first
Was Mortal Kombat X released for PlayStation 5?     -> no matching game (not in the dataset)
Reload check                                        -> reopens the database from disk: 15 documents
```

**Part 2 - agent.** For each query the output shows the steps of the agent (retrieval, evaluation with its reasoning, web search when needed), then the final report:

```
Session "demo"
  When was Pokémon Gold and Silver released?          -> answered from the local dataset
  Which one was the first 3D platformer Mario game?   -> answered from the local dataset
  Was Mortal Kombat X released for PlayStation 5?     -> not in the dataset: web search, cites URLs

Session "conversation" (conversation state)
  Who published Grand Theft Auto: San Andreas?        -> local dataset
  On which platform was it released, and in what year?  -> "it" resolved to GTA: San Andreas
  What is that publisher working on right now?        -> resolved to Rockstar Games, needs current info: web search

New agent, new session (long-term memory)
  Was Mortal Kombat X released for PlayStation 5?     -> answered from the long-term memory, no new web search

Structured output
  The last report printed as JSON
```

## Project Instructions

### Part 1 - RAG pipeline

- `lib/game_store.py` - `GameVectorStore`, a reusable manager for the vector store. It wraps the persistent client (`chromadb.PersistentClient(path="chromadb")`), the OpenAI embedding function (key from `OPENAI_API_KEY`, endpoint from `OPENAI_BASE_URL`) and the `udaplay` collection.
  - `load_games()` reads the JSON files in `games/` and upserts them. The embedded text includes platform, name, year, genre, publisher and description.
  - `search()` runs a semantic search and returns the games with their distance.
- The notebook indexes the 15 games, runs the semantic search queries and reloads the database from disk.

### Part 2 - Agent

**Tools** (functions decorated with `@tool` from `lib/tooling.py`):

| Tool | What it does |
|---|---|
| `retrieve_game` | Semantic search in the `udaplay` collection |
| `evaluate_retrieval` | LLM judge: are the documents enough to answer? Returns `useful`, `description` and `confidence` |
| `game_web_search` | Web search with Tavily: titles, URLs and page content |
| `search_memory` / `store_memory` | Read and write the long-term memory (optional extension) |

**State machine.** `UdaPlayAgent` is built on `lib/state_machine.py`, with the tools as predefined nodes, so every question follows the same path:

```
contextualize -> retrieve_game -> search_memory -> evaluate_retrieval
    -> game_web_search   (only if the documents are not useful, or confidence < 0.7)
    -> generate_answer
    -> store_memory      (only if the answer came from the web)
```

**Conversation state.** Each session keeps its runs in a `ShortTermMemory`. The `contextualize` step rewrites follow-up questions ("who published it?") into standalone ones, using the previous turns.

**Reports.** `generate_answer` returns a structured `UdaPlayReport` with:

- the answer, with inline citations like `[S1]`;
- the confidence of the answer and the retrieval evaluation;
- whether the web was used;
- the cited sources, with their origin (local dataset, long-term memory or web) and URLs.

The prompt only allows citing a source for facts it actually contains, and prefers authoritative sources. Reports are printed in a readable format and are also available as JSON.

### Optional extensions

- **Long-term memory**: answers learned from the web are saved in a second persistent ChromaDB collection, `udaplay_memory`. A new agent instance, in a new session, finds them without searching the web again.
- **Structured output**: every answer is available both in natural language and as JSON (`UdaPlayReport.model_dump_json()`).

## Built With

* [ChromaDB](https://www.trychroma.com/) - Persistent vector database for the games and the long-term memory
* [OpenAI API](https://platform.openai.com/docs) - `gpt-4o-mini` for evaluation and answers, OpenAI embeddings for semantic search
* [Tavily](https://tavily.com/) - Web search API for the fallback
* [Pydantic](https://docs.pydantic.dev/) - Structured outputs (evaluation and reports)
* [python-dotenv](https://github.com/theskumar/python-dotenv) - Loads the API keys from `.env`
* [Jupyter](https://jupyter.org/) - Notebooks for both parts of the project
