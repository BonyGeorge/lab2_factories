import os, json
from typing import Dict, Any
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email

class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
    
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
    
    def classify_email(self, email: Email, mode="topic") -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
        
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
        
        # Step 2: Classify using features
        predicted_topic = self.model.predict(features, mode=mode)
        topic_scores = self.model.get_topic_scores(features)
        
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }
    
    def add_topic(self, topic: Dict[str, Any]):
        """Add new Topic"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                existing_topics = json.load(f)
        else:
            existing_topics = {}
        
        existing_topics.update(topic)
        
        with open(data_file, 'w') as f:
            json.dump(existing_topics, f, indent=2)
            
    def add_email(self, email: Dict[str, Any]):
        """Add new Email"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                existing_emails = json.load(f, strict=False)
        else:
            existing_emails = []
        
        existing_emails.append(email)
        
        with open(data_file, 'w') as f:
            json.dump(existing_emails, f, indent=2)
    
    def get_all_emails(self) -> list:
        """Retrieve all emails"""
        data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')
        
        with open(data_file, 'r') as f:
            emails = json.load(f)
        
        return emails