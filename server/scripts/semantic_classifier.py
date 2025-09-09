#!/usr/bin/env python3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from utils.decorators import track_performance
from utils import config

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SemanticClassifier:
    def __init__(self):
        self.model_name = config.get_str('semantic', 'model_name', 'all-MiniLM-L6-v2')
        self.human_threshold = config.get_float('semantic', 'human_threshold', 0.6)
        self.animal_threshold = config.get_float('semantic', 'animal_threshold', 0.5)
        self.similarity_metric = config.get_str('semantic', 'similarity_metric', 'cosine')
        
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
        
        self.embedding_cache = {}
        
        self._initialize_model()
        self._precompute_reference_embeddings()
    
    def _initialize_model(self):
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None
    
    def _precompute_reference_embeddings(self):
        self.human_embeddings = None
        self.animal_embeddings = None
        if self.model is not None:
            try:
                self.human_embeddings = self.model.encode(self.human_terms)
                self.animal_embeddings = self.model.encode(self.animal_terms)
                self.human_centroid = np.mean(self.human_embeddings, axis=0)
                self.animal_centroid = np.mean(self.animal_embeddings, axis=0)
            except Exception:
                self.human_embeddings = None
                self.animal_embeddings = None
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            embedding = self.model.encode([text])[0]
            self.embedding_cache[text] = embedding
            return embedding
        except Exception:
            return None
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if SKLEARN_AVAILABLE:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        else:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def _compute_max_similarity_to_category(self, text_embedding: np.ndarray, 
                                          category_embeddings: np.ndarray) -> float:
        max_similarity = 0.0
        for ref_embedding in category_embeddings:
            similarity = self._compute_similarity(text_embedding, ref_embedding)
            max_similarity = max(max_similarity, similarity)
        return max_similarity
    
    def _compute_centroid_similarity(self, text_embedding: np.ndarray, 
                                   centroid: np.ndarray) -> float:
        return self._compute_similarity(text_embedding, centroid)
    
    def _fallback_classification(self, class_name: str) -> str:
        class_name_lower = class_name.lower()
        for term in self.human_terms:
            if term in class_name_lower:
                return 'human'
        for term in self.animal_terms:
            if term in class_name_lower:
                return 'animal'
        return 'other'
    
    @track_performance
    def classify_class_name(self, class_name: str) -> Dict[str, Any]:
        result = {
            'class_name': class_name,
            'semantic_category': 'other',
            'confidence': 0.0,
            'human_similarity': 0.0,
            'animal_similarity': 0.0,
            'method': 'fallback'
        }
        
        if (self.model is None or 
            self.human_embeddings is None or 
            self.animal_embeddings is None):
            result['semantic_category'] = self._fallback_classification(class_name)
            result['method'] = 'fallback'
            result['confidence'] = 1.0 if result['semantic_category'] != 'other' else 0.0
            return result
        
        text_embedding = self._get_embedding(class_name)
        if text_embedding is None:
            result['semantic_category'] = self._fallback_classification(class_name)
            result['method'] = 'fallback'
            return result
        
        human_similarity = self._compute_max_similarity_to_category(
            text_embedding, self.human_embeddings
        )
        animal_similarity = self._compute_max_similarity_to_category(
            text_embedding, self.animal_embeddings
        )
        
        human_centroid_sim = self._compute_centroid_similarity(text_embedding, self.human_centroid)
        animal_centroid_sim = self._compute_centroid_similarity(text_embedding, self.animal_centroid)
        
        final_human_sim = max(human_similarity, human_centroid_sim)
        final_animal_sim = max(animal_similarity, animal_centroid_sim)
        
        category = 'other'
        confidence = 0.0
        
        if final_human_sim >= self.human_threshold and final_human_sim > final_animal_sim + 0.1:
            category = 'human'
            confidence = final_human_sim
        elif final_animal_sim >= self.animal_threshold and final_animal_sim > final_human_sim + 0.1:
            category = 'animal'
            confidence = final_animal_sim
        
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
        classified_objects = []
        for obj in objects:
            class_name = obj.get('class_name', 'unknown')
            classification = self.classify_class_name(class_name)
            enhanced_obj = obj.copy()
            enhanced_obj.update({
                'original_class_name': class_name,
                'class_name': classification['semantic_category'],
                'semantic_category': classification['semantic_category'],
                'semantic_confidence': classification['confidence'],
                'human_similarity': classification['human_similarity'],
                'animal_similarity': classification['animal_similarity'],
                'classification_method': classification['method']
            })
            classified_objects.append(enhanced_obj)
        return classified_objects
    
    def get_classification_statistics(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            category = obj.get('semantic_category', 'other')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            method = obj.get('classification_method', 'unknown')
            if method not in methods:
                methods[method] = 0
            methods[method] += 1
            
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
        if human_threshold is not None:
            self.human_threshold = human_threshold
        if animal_threshold is not None:
            self.animal_threshold = animal_threshold
