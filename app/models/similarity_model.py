import os
import json
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self, stored_emails):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()

        self.stored_emails = stored_emails
    
    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        with open(data_file, 'r') as f:
            return json.load(f)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings
    
    def predict(self, features: dict, mode: str = "topic") -> str:
        email_embedding = features.get("email_embeddings_average_embedding")
        if email_embedding is None:
            return None

        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        if mode == "topic":
            scores = {
                topic: self._calculate_topic_score(features, topic)
                for topic in self.topics
            }
            return max(scores, key=scores.get)

        if mode == "nearest":
            return self._predict_by_nearest_email(email_embedding)
        
        else:
            raise ValueError("Mode must be 'topic' or 'nearest'")

    def _predict_by_nearest_email(self, email_embedding: np.ndarray) -> str:
        best_score = -1
        best_label = None

        for stored in self.stored_emails:
            stored_embedding = np.array(stored["embedding"])
            label = stored.get("ground_truth")
            if label is None:
                continue

            dot = np.dot(email_embedding, stored_embedding)
            norm = np.linalg.norm(email_embedding) * np.linalg.norm(stored_embedding)
            if norm == 0:
                continue

            sim = dot / norm
            if sim > best_score:
                best_score = sim
                best_label = label

        return best_label
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate cosine similarity between email and topic embeddings"""
        # Get email embedding from features (now a list/array)
        email_embedding = features.get("email_embeddings_average_embedding", None)

        if email_embedding is None:
            return 0.0

        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        # Get pre-computed topic embedding
        topic_embedding = self.topic_embeddings[topic]

        # Calculate cosine similarity
        # cosine_similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(email_embedding, topic_embedding)
        email_norm = np.linalg.norm(email_embedding)
        topic_norm = np.linalg.norm(topic_embedding)

        if email_norm == 0 or topic_norm == 0:
            return 0.0

        cosine_similarity = dot_product / (email_norm * topic_norm)

        # Cosine similarity is between -1 and 1, but for text it's usually positive
        # Normalize to 0-1 range for better interpretability
        normalized_score = (cosine_similarity + 1) / 2

        return float(normalized_score)
    
    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}
    
    def add_topic(self, topic: Dict[str, Any]):
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        
        with open(data_file, 'r') as f:
            current_topics = json.load(f)

        current_topics.update(topic)

        with open(data_file, 'w') as f:
            json.dump(current_topics, f, indent=4)

        self.topic_data = current_topics
        self.topics = list(self.topic_data.keys())
        self.topic_embeddings = self._compute_topic_embeddings()
