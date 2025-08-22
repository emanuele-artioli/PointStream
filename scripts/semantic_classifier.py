#!/usr/bin/env python3
"""
Semantic Classifier Component

This component handles semantic classification of object class names into 
'human', 'animal', or 'other' categories using embedding-based similarity.

Uses sentence transformers to create embeddings of class names and compare
them to reference embeddings for each category.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .decorators import track_performance
from . import config

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logging.info("SentenceTransformers available for semantic classification")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available, using fallback classification")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, using numpy cosine similarity")


class SemanticClassifier:
    """Semantic classifier using embedding-based similarity."""
    
    def __init__(self):
        """Initialize the semantic classifier with reference embeddings."""
        # Load configuration
        self.model_name = config.get_str('semantic', 'model_name', 'all-MiniLM-L6-v2')
        self.human_threshold = config.get_float('semantic', 'human_threshold', 0.6)
        self.animal_threshold = config.get_float('semantic', 'animal_threshold', 0.5)
        self.similarity_metric = config.get_str('semantic', 'similarity_metric', 'cosine')
        
        # Reference terms for each category
        self.human_terms = [
            'person', 'human', 'man', 'woman', 'child', 'baby', 'people',
            'boy', 'girl', 'adult', 'individual', 'someone', 'anybody',
            'pedestrian', 'passenger', 'worker', 'player', 'athlete'
        ]
        
        self.animal_terms = [
            'dog', 'cat', 'horse', 'cow', 'sheep', 'pig', 'goat',
            'elephant', 'bear', 'lion', 'tiger', 'zebra', 'giraffe',
            'bird', 'chicken', 'duck', 'goose', 'turkey', 'eagle',
            'fish', 'shark', 'whale', 'dolphin', 'salmon', 'goldfish',
            'rabbit', 'mouse', 'rat', 'hamster', 'guinea pig',
            'animal', 'mammal', 'creature', 'beast', 'wildlife',
            'pet', 'livestock', 'fauna'
        ]
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        
        # Initialize model and reference embeddings
        self._initialize_model()
        self._precompute_reference_embeddings()
        
        logging.info("Semantic classifier initialized")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Human threshold: {self.human_threshold}")
        logging.info(f"Animal threshold: {self.animal_threshold}")
        logging.info(f"Reference terms: {len(self.human_terms)} human, {len(self.animal_terms)} animal")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logging.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                logging.error(f"Failed to load SentenceTransformer model: {e}")
                self.model = None
        
        if self.model is None:
            logging.warning("SentenceTransformer not available, using fallback classification")
    
    def _precompute_reference_embeddings(self):
        """Precompute embeddings for reference terms."""
        self.human_embeddings = None
        self.animal_embeddings = None
        
        if self.model is not None:
            try:
                # Compute embeddings for reference terms
                self.human_embeddings = self.model.encode(self.human_terms)
                self.animal_embeddings = self.model.encode(self.animal_terms)
                
                # Compute average embeddings for each category
                self.human_centroid = np.mean(self.human_embeddings, axis=0)
                self.animal_centroid = np.mean(self.animal_embeddings, axis=0)
                
                logging.info("Reference embeddings computed successfully")
                
            except Exception as e:
                logging.error(f"Failed to compute reference embeddings: {e}")
                self.human_embeddings = None
                self.animal_embeddings = None
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text, using cache if available."""
        if self.model is None:
            return None
        
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.model.encode([text])[0]
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logging.error(f"Failed to compute embedding for '{text}': {e}")
            return None
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings."""
        if SKLEARN_AVAILABLE:
            # Use sklearn's cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        else:
            # Use numpy to compute cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def _compute_max_similarity_to_category(self, text_embedding: np.ndarray, 
                                          category_embeddings: np.ndarray) -> float:
        """Compute maximum similarity to any term in a category."""
        max_similarity = 0.0
        
        for ref_embedding in category_embeddings:
            similarity = self._compute_similarity(text_embedding, ref_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _compute_centroid_similarity(self, text_embedding: np.ndarray, 
                                   centroid: np.ndarray) -> float:
        """Compute similarity to category centroid."""
        return self._compute_similarity(text_embedding, centroid)
    
    def _fallback_classification(self, class_name: str) -> str:
        """Fallback classification when embeddings are not available."""
        class_name_lower = class_name.lower()
        
        # Simple keyword matching as fallback
        for term in self.human_terms:
            if term in class_name_lower:
                return 'human'
        
        for term in self.animal_terms:
            if term in class_name_lower:
                return 'animal'
        
        return 'other'
    
    @track_performance
    def classify_class_name(self, class_name: str) -> Dict[str, Any]:
        """
        Classify a single class name into semantic categories.
        
        Args:
            class_name: Object class name from detection
            
        Returns:
            Dictionary with classification results
        """
        result = {
            'class_name': class_name,
            'semantic_category': 'other',
            'confidence': 0.0,
            'human_similarity': 0.0,
            'animal_similarity': 0.0,
            'method': 'fallback'
        }
        
        # Use fallback if embeddings not available
        if (self.model is None or 
            self.human_embeddings is None or 
            self.animal_embeddings is None):
            result['semantic_category'] = self._fallback_classification(class_name)
            result['method'] = 'fallback'
            result['confidence'] = 1.0 if result['semantic_category'] != 'other' else 0.0
            return result
        
        # Get embedding for the class name
        text_embedding = self._get_embedding(class_name)
        if text_embedding is None:
            result['semantic_category'] = self._fallback_classification(class_name)
            result['method'] = 'fallback'
            return result
        
        # Compute similarities to each category
        human_similarity = self._compute_max_similarity_to_category(
            text_embedding, self.human_embeddings
        )
        animal_similarity = self._compute_max_similarity_to_category(
            text_embedding, self.animal_embeddings
        )
        
        # Also compute centroid similarities for additional confidence
        human_centroid_sim = self._compute_centroid_similarity(text_embedding, self.human_centroid)
        animal_centroid_sim = self._compute_centroid_similarity(text_embedding, self.animal_centroid)
        
        # Use the maximum of max similarity and centroid similarity for each category
        final_human_sim = max(human_similarity, human_centroid_sim)
        final_animal_sim = max(animal_similarity, animal_centroid_sim)
        
        # Classify based on thresholds and relative similarities
        category = 'other'
        confidence = 0.0
        
        # Require both threshold satisfaction AND clear winner
        if final_human_sim >= self.human_threshold and final_human_sim > final_animal_sim + 0.1:
            category = 'human'
            confidence = final_human_sim
        elif final_animal_sim >= self.animal_threshold and final_animal_sim > final_human_sim + 0.1:
            category = 'animal'
            confidence = final_animal_sim
        
        # If no clear winner above thresholds, classify as 'other'
        
        result.update({
            'semantic_category': category,
            'confidence': confidence,
            'human_similarity': final_human_sim,
            'animal_similarity': final_animal_sim,
            'method': 'embedding'
        })
        
        return result
    
    @track_performance
    def classify_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple objects and add semantic categories.
        
        Args:
            objects: List of object dictionaries with 'class_name' field
            
        Returns:
            List of objects with added semantic classification
        """
        classified_objects = []
        
        for obj in objects:
            class_name = obj.get('class_name', 'unknown')
            
            # Get classification
            classification = self.classify_class_name(class_name)
            
            # Add classification to object
            enhanced_obj = obj.copy()
            enhanced_obj.update({
                'original_class_name': class_name,
                'class_name': classification['semantic_category'],  # Replace with semantic category
                'semantic_category': classification['semantic_category'],
                'semantic_confidence': classification['confidence'],
                'human_similarity': classification['human_similarity'],
                'animal_similarity': classification['animal_similarity'],
                'classification_method': classification['method']
            })
            
            classified_objects.append(enhanced_obj)
        
        return classified_objects
    
    def get_classification_statistics(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about classification results.
        
        Args:
            objects: List of classified objects
            
        Returns:
            Dictionary with classification statistics
        """
        if not objects:
            return {
                'total_objects': 0,
                'categories': {},
                'methods': {},
                'average_confidence': 0.0
            }
        
        categories = {}
        methods = {}
        confidences = []
        
        for obj in objects:
            # Count categories
            category = obj.get('semantic_category', 'other')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            # Count methods
            method = obj.get('classification_method', 'unknown')
            if method not in methods:
                methods[method] = 0
            methods[method] += 1
            
            # Collect confidences
            confidence = obj.get('semantic_confidence', 0.0)
            confidences.append(confidence)
        
        return {
            'total_objects': len(objects),
            'categories': categories,
            'methods': methods,
            'average_confidence': np.mean(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0
        }
    
    def update_thresholds(self, human_threshold: float = None, 
                         animal_threshold: float = None):
        """
        Update classification thresholds dynamically.
        
        Args:
            human_threshold: New threshold for human classification
            animal_threshold: New threshold for animal classification
        """
        if human_threshold is not None:
            self.human_threshold = human_threshold
            logging.info(f"Updated human threshold to {human_threshold}")
        
        if animal_threshold is not None:
            self.animal_threshold = animal_threshold
            logging.info(f"Updated animal threshold to {animal_threshold}")
