from openai import OpenAI
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import Any

_ = load_dotenv()

class EmbeddingManager:
    def __init__(self, collection_name: str = "math_problems"):
        """Initialize the embedding manager with OpenAI client and ChromaDB."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-3-small",
        )
        
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=self.openai_ef)
        
    def _create_feature_text(self, features: dict[str, Any]) -> str:
        """Convert features dictionary into a text description."""
        general_type = features.get("general_type", "unknown")
        specific_topics = ", ".join(features.get("specific_topics", []))
        operations = ", ".join(features.get("operations", []))
        theorems = ", ".join(features.get("theorems", []))
        difficulty = features.get("difficulty_level", "unknown")
        
        feature_text = f"Type: {general_type}. Topics: {specific_topics}. "
        feature_text += f"Operations: {operations}. Theorems: {theorems}. "
        feature_text += f"Difficulty: {difficulty}."
        
        return feature_text
    
    def load_training_examples(self, file_path: str = "training_examples.json") -> list[dict[str, Any]]:
        """Load training examples from a JSON file."""
        try:
            with open(file_path, "r") as f:
                examples = json.load(f)
            return examples
        except Exception as e:
            print(f"Error loading training examples: {e}")
            return []
        
    def add_examples_to_db(self, examples: list[dict[str, Any]]) -> None:
        """Add training examples to the vector database."""
        ids = []
        documents = []
        metadatas = []
        
        for example in examples:
            idx = str(example.get("idx", ""))
            question = example.get("question", "")
            features = example.get("features", {})
            solution = example.get("modified_solution", "")
            
            # Create feature text for embedding
            feature_text = self._create_feature_text(features)
            
            # Add to lists for batch processing
            ids.append(idx)
            documents.append(feature_text)
            metadatas.append({
                "question": question,
                "solution": solution,
                "general_type": features.get("general_type", ""),
                "difficulty": features.get("difficulty_level", "")
            })
        
        # Add to ChromaDB collection
        if ids:
            # Check if IDs already exist and remove them first
            existing_ids = set(self.collection.get(ids=ids, include=[])["ids"])
            if existing_ids:
                self.collection.delete(ids=list(existing_ids))
            
            # Add new embeddings
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} examples to the vector database.")
            
    def search_similar_problems(self, question: str, features: dict[str, Any], n_results: int = 3) -> list[dict[str, Any]]:
        """Search for similar problems based on question features."""
        feature_text = self._create_feature_text(features)
        
        # Query the collection
        results = self.collection.query(
            query_texts=[feature_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "question": results["metadatas"][0][i]["question"],
                    "solution": results["metadatas"][0][i]["solution"],
                    "general_type": results["metadatas"][0][i]["general_type"],
                    "difficulty": results["metadatas"][0][i]["difficulty"]
                })
        
        return formatted_results
    
    def extract_features_from_question(self, question: str) -> dict[str, Any]:
        """Extract features from a new question using OpenAI."""
        prompt = f"""
        Analyze this math problem to extract features.
        
        Question: {question}
        
        Extract the following information in JSON format:
        1. general_type: General math category (e.g., algebra, geometry, calculus, probability, etc.)
        2. specific_topics: List of specific math topics involved (e.g., quadratic equations, triangles, derivatives, etc.)
        3. operations: List of mathematical operations likely needed in solution
        4. theorems: List of theorems, formulas or principles likely needed
        5. difficulty_level: Estimated difficulty (easy, medium, hard)
        
        Return ONLY the JSON object without any additional text.
        """
        
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a math analysis assistant that extracts structured features from math problems."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            result_text = completion.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            result = json.loads(result_text.strip())
            return result
        except:
            return {
                "general_type": "unknown",
                "specific_topics": [],
                "operations": [],
                "theorems": [],
                "difficulty_level": "unknown"
            }
            
            
embedding_manager = EmbeddingManager()
examples = embedding_manager.load_training_examples()

# if examples:
#     embedding_manager.add_examples_to_db(examples)
#     print(f"Successfully processed {len(examples)} training examples.")