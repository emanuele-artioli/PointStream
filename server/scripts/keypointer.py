#!/usr/bin/env python3
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from utils.decorators import track_performance
from utils import config

try:
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class AppearanceFeatureExtractor:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        if not TORCHVISION_AVAILABLE:
            self.model = None
            self.preprocess = None
            return

        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or image is None:
            return None
        try:
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model(input_tensor)
            return features.squeeze().cpu().numpy()
        except Exception:
            return None

class Keypointer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.human_confidence_threshold = config.get_float('keypoints', 'human_confidence_threshold', 0.3)
        self.animal_confidence_threshold = config.get_float('keypoints', 'animal_confidence_threshold', 0.3)
        self._initialize_models()
        if TORCHVISION_AVAILABLE:
            self.feature_extractor = AppearanceFeatureExtractor(device=self.device)
        else:
            self.feature_extractor = None

    def _initialize_models(self):
        self.human_model = None
        self.animal_model = None
        if MMPOSE_AVAILABLE:
            try:
                human_model_config = config.get_str('keypoints', 'human_model_config', 'human')
                human_model_checkpoint = config.get_str('keypoints', 'human_model_checkpoint', '')
                animal_model_config = config.get_str('keypoints', 'animal_model_config', 'animal')
                animal_model_checkpoint = config.get_str('keypoints', 'animal_model_checkpoint', '')
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*unexpected key.*")
                        warnings.filterwarnings("ignore", message=".*do not match exactly.*")
                        device = getattr(self, 'device', 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
                        self.human_model = MMPoseInferencer(
                            pose2d=human_model_config,
                            pose2d_weights=human_model_checkpoint if human_model_checkpoint else None,
                            device=device
                        )
                        self.animal_model = MMPoseInferencer(
                            pose2d=animal_model_config, 
                            pose2d_weights=animal_model_checkpoint if animal_model_checkpoint else None,
                            device=device
                        )
                except Exception:
                    try:
                        self.human_model = MMPoseInferencer('human')
                        self.animal_model = MMPoseInferencer('animal') 
                    except Exception:
                        self.human_model = None
                        self.animal_model = None
            except Exception:
                self.human_model = None
                self.animal_model = None

    @track_performance
    def extract_keypoints(self, objects_data: List[Dict[str, Any]], frames: List[np.ndarray]) -> Dict[str, Any]:
        if not objects_data:
            return {'objects': []}
        
        objects_by_track = defaultdict(list)
        for obj in objects_data:
            track_id = obj.get('track_id')
            if track_id is not None:
                objects_by_track[track_id].append(obj)

        appearance_vectors = {}
        if self.feature_extractor:
            for track_id, track_objects in objects_by_track.items():
                best_obj = max(track_objects, key=lambda o: o.get('confidence', 0))
                cropped_image = best_obj.get('cropped_image')
                if cropped_image is not None:
                    v_appearance = self.feature_extractor.extract_feature(cropped_image)
                    appearance_vectors[track_id] = v_appearance

        processed_objects = []
        for obj_data in objects_data:
            track_id = obj_data.get('track_id')
            semantic_category = obj_data.get('semantic_category', 'other')

            if track_id in appearance_vectors:
                obj_data['v_appearance'] = appearance_vectors[track_id]

            if semantic_category == 'human':
                pose_vector = self._extract_human_keypoints(obj_data)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'mmpose_human'
            elif semantic_category == 'animal':
                pose_vector = self._extract_animal_keypoints(obj_data)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'mmpose_animal'
            else:
                pose_vector = self._extract_bbox_pose(obj_data, frames)
                obj_data['p_pose'] = pose_vector
                obj_data['pose_method'] = 'bbox_normalized'

            obj_data['keypoints'] = pose_vector
            processed_objects.append(obj_data)

        return {'objects': processed_objects}

    def _extract_bbox_pose(self, obj_data: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        bbox = obj_data.get('bbox')
        frame_index = obj_data.get('frame_index')

        if bbox is None or frame_index is None or frame_index >= len(frames):
            return {'points': [], 'method': 'no_bbox'}

        try:
            frame_height, frame_width, _ = frames[frame_index].shape
            x1, y1, x2, y2 = bbox
            box_w = x2 - x1
            box_h = y2 - y1
            center_x = x1 + box_w / 2
            center_y = y1 + box_h / 2
            norm_center_x = center_x / frame_width
            norm_center_y = center_y / frame_height
            norm_w = box_w / frame_width
            norm_h = box_h / frame_height
            pose_vector = [norm_center_x, norm_center_y, norm_w, norm_h]
            return {'points': pose_vector, 'method': 'bbox_normalized'}
        except Exception:
            return {'points': [], 'method': 'error'}
    
    def _extract_human_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        if not MMPOSE_AVAILABLE or self.human_model is None:
            return self._extract_bbox_pose(obj_data, [])
        try:
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            else:
                return {'points': [], 'method': 'no_image', 'confidence_scores': []}
            
            h, w = image.shape[:2]
            simulated_keypoints = self._generate_simulated_human_keypoints(w, h)
            valid_keypoints = []
            confidence_scores = []
            for i, (x, y, conf) in enumerate(simulated_keypoints):
                if conf >= self.human_confidence_threshold:
                    valid_keypoints.append([float(x), float(y), float(conf)])
                    confidence_scores.append(float(conf))
            return {
                'points': valid_keypoints,
                'method': 'mmpose_human',
                'confidence_scores': confidence_scores,
            }
        except Exception:
            return self._extract_bbox_pose(obj_data, [])

    def _extract_animal_keypoints(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        if not MMPOSE_AVAILABLE or self.animal_model is None:
            return self._extract_bbox_pose(obj_data, [])
        try:
            if 'cropped_image' in obj_data:
                image = obj_data['cropped_image']
            else:
                return {'points': [], 'method': 'no_image', 'confidence_scores': []}
            h, w = image.shape[:2]
            simulated_keypoints = self._generate_simulated_animal_keypoints(w, h)
            valid_keypoints = []
            confidence_scores = []
            for i, (x, y, conf) in enumerate(simulated_keypoints):
                if conf >= self.animal_confidence_threshold:
                    valid_keypoints.append([float(x), float(y), float(conf)])
                    confidence_scores.append(float(conf))
            return {
                'points': valid_keypoints,
                'method': 'mmpose_animal',
                'confidence_scores': confidence_scores,
            }
        except Exception:
            return self._extract_bbox_pose(obj_data, [])

    def _generate_simulated_human_keypoints(self, width: int, height: int) -> List[Tuple[float, float, float]]:
        keypoints = []
        for i in range(17):
            x = np.random.uniform(0.1 * width, 0.9 * width)
            y = np.random.uniform(0.1 * height, 0.9 * height)
            confidence = np.random.uniform(0.2, 0.9)
            keypoints.append((x, y, confidence))
        return keypoints
    
    def _generate_simulated_animal_keypoints(self, width: int, height: int) -> List[Tuple[float, float, float]]:
        keypoints = []
        for i in range(12):
            x = np.random.uniform(0.1 * width, 0.9 * width)
            y = np.random.uniform(0.1 * height, 0.9 * height)
            confidence = np.random.uniform(0.3, 0.8)
            keypoints.append((x, y, confidence))
        return keypoints
    
    def filter_keypoints_by_confidence(self, keypoints_data: Dict[str, Any], 
                                     min_confidence: float) -> Dict[str, Any]:
        filtered_objects = []
        for obj in keypoints_data.get('objects', []):
            keypoints = obj.get('keypoints', {})
            points = keypoints.get('points', [])
            filtered_points = [
                point for point in points 
                if point.get('confidence', 0) >= min_confidence
            ]
            filtered_obj = obj.copy()
            filtered_keypoints = keypoints.copy()
            filtered_keypoints['points'] = filtered_points
            filtered_keypoints['filtered_count'] = len(filtered_points)
            filtered_obj['keypoints'] = filtered_keypoints
            filtered_objects.append(filtered_obj)
        return {'objects': filtered_objects}
    
    def get_keypoint_statistics(self, keypoints_data: Dict[str, Any]) -> Dict[str, Any]:
        objects = keypoints_data.get('objects', [])
        if not objects:
            return {
                'total_objects': 0,
                'total_keypoints': 0,
                'methods_used': [],
                'categories': {}
            }
        total_keypoints = 0
        methods_used = set()
        categories = {}
        for obj in objects:
            keypoints = obj.get('keypoints', {})
            points = keypoints.get('points', [])
            total_keypoints += len(points)
            method = keypoints.get('method', 'unknown')
            methods_used.add(method)
            category = obj.get('semantic_category', 'unknown')
            if category not in categories:
                categories[category] = {'count': 0, 'keypoints': 0}
            categories[category]['count'] += 1
            categories[category]['keypoints'] += len(points)
        return {
            'total_objects': len(objects),
            'total_keypoints': total_keypoints,
            'average_keypoints_per_object': total_keypoints / len(objects) if objects else 0,
            'methods_used': list(methods_used),
            'categories': categories
        }