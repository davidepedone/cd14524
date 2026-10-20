import os
import json
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection as ChromaCollection


class GameVectorStore:
    """
    Reusable manager for the persistent UdaPlay games vector store.

    Wraps the ChromaDB persistent client, the OpenAI embedding function and the
    games collection, so Part 1 (indexing) and Part 2 (the agent) open the
    database the same way: same path, same collection name, same embeddings.

    The embedding function reads the key from OPENAI_API_KEY and the endpoint
    from OPENAI_BASE_URL, so load the .env file before creating the store.
    """

    def __init__(self, path: str = "chromadb", collection_name: str = "udaplay",
                 create: bool = True):
        """
        Args:
            path: Folder of the persistent ChromaDB database
            collection_name: Name of the games collection
            create: If True, create the collection when missing (Part 1).
                If False, the collection must already exist (Part 2): a
                missing collection raises an error instead of silently
                returning an empty store.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key_env_var="OPENAI_API_KEY",
            api_base=os.getenv("OPENAI_BASE_URL"),
        )
        if create:
            self.collection: ChromaCollection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        else:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )

    def __repr__(self) -> str:
        return f"GameVectorStore(collection='{self.collection.name}', documents={self.count()})"

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def format_game(game: Dict[str, Any]) -> str:
        """Text that gets embedded for a game.

        Genre and Publisher are included so questions like
        "Who published ...?" can be matched semantically.
        """
        return (
            f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - "
            f"Genre: {game['Genre']}. Publisher: {game['Publisher']}. "
            f"{game['Description']}"
        )

    def load_games(self, data_dir: str = "games") -> int:
        """
        Index every JSON file in data_dir, using the file name (like 001) as id.

        Uses upsert, so loading the same folder again updates the documents
        instead of creating duplicates.

        Returns:
            Number of games indexed
        """
        ids, documents, metadatas = [], [], []
        for file_name in sorted(os.listdir(data_dir)):
            if not file_name.endswith(".json"):
                continue
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
                game = json.load(f)
            ids.append(os.path.splitext(file_name)[0])
            documents.append(self.format_game(game))
            metadatas.append(game)

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search over the games.

        Returns:
            One dict per game, most similar first, with the game fields
            (Name, Platform, Genre, Publisher, Description, YearOfRelease),
            its id and the distance from the query (lower is closer).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        return [
            {"id": doc_id, **metadata, "distance": round(distance, 4)}
            for doc_id, metadata, distance in zip(
                results["ids"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
