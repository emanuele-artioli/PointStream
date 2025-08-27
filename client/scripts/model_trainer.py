#!/usr/bin/env python3
"""
Model Trainer

This module trains the generative models (cGAN) for different object categories.
Handles training data preparation, model training, and checkpoint management.
"""

import logging
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import os

try:
    from utils.decorators import track_performance
    from utils import config
    from client.models.human_cgan import HumanCGAN
    from client.models.animal_cgan import AnimalCGAN
    from client.models.other_cgan import OtherCGAN
    
    # Optional imports for advanced features
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities or models: {e}")
    raise


def collate_fn(batch):
    """Custom collate function to filter out None values."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return (None, None, None)
    return torch.utils.data.dataloader.default_collate(batch)


class ObjectDataset(Dataset):
    """Dataset for training object generation models based on a reference image."""
    
    def __init__(self, objects_data: Dict[str, List[Dict[str, Any]]], category: str,
                 input_size: int = 256, augment: bool = True):
        """
        Initialize object dataset.
        
        Args:
            objects_data: Dict mapping object_id to a list of its appearances.
            category: Object category (human, animal, other)
            input_size: Input image size
            augment: Whether to apply data augmentation
        """
        self.category = category
        self.input_size = input_size
        self.augment = augment
        self.training_pairs = []
        
        # Create training pairs (reference_object, target_object)
        for object_id, appearances in objects_data.items():
            # Filter for valid objects first
            valid_appearances = []
            for obj in appearances:
                if self._validate_object(obj):
                    valid_appearances.append(obj)

            # Need at least one reference and one target
            if len(valid_appearances) < 2:
                continue

            ref_obj = valid_appearances[0]

            # Use all subsequent valid appearances as targets
            for i in range(1, len(valid_appearances)):
                target_obj = valid_appearances[i]
                self.training_pairs.append((ref_obj, target_obj))
        
        logging.info(f"📚 {category.capitalize()} dataset: {len(self.training_pairs)} training pairs created from {len(objects_data)} unique objects.")

    def _validate_object(self, obj: Dict[str, Any]) -> bool:
        """Validate that object has required data and the image exists."""
        if 'keypoints' not in obj or not obj['keypoints']:
            return False

        image_path = obj.get('cropped_image')
        if image_path is None:
            return False
        
        if isinstance(image_path, str) and not os.path.exists(image_path):
            logging.warning(f"Image path not found, skipping: {image_path}")
            return False

        return True

    def __len__(self) -> int:
        return len(self.training_pairs)

    def _load_image(self, image_data: Any) -> Optional[np.ndarray]:
        """Load image from path or use if already a numpy array."""
        if isinstance(image_data, str):
            img = cv2.imread(image_data)
            if img is None:
                logging.error(f"Failed to read image: {image_data}")
            return img
        return image_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training item. Assumes that data has been pre-validated in __init__.
        """
        ref_obj, target_obj = self.training_pairs[idx]
        
        # --- Process Reference Image ---
        ref_image = self._load_image(ref_obj['cropped_image'])
        ref_image = cv2.resize(ref_image, (self.input_size, self.input_size))
        ref_tensor = torch.from_numpy(ref_image).float().permute(2, 0, 1) / 127.5 - 1.0

        # --- Process Target Image ---
        target_image = self._load_image(target_obj['cropped_image'])
        target_image = cv2.resize(target_image, (self.input_size, self.input_size))
        
        # --- Process Feature Map (from target keypoints) ---
        keypoints = target_obj['keypoints']
        if self.category in ['human', 'animal']:
            feature_map = self._create_pose_map(keypoints)
        else:
            feature_map = self._create_feature_map(keypoints)
        
        # --- Data Augmentation (on target and feature map) ---
        if self.augment:
            target_image, feature_map = self._apply_augmentation(target_image, feature_map)
        
        # --- Convert to Tensors and Normalize ---
        target_tensor = torch.from_numpy(target_image).float().permute(2, 0, 1) / 127.5 - 1.0
        feature_tensor = torch.from_numpy(feature_map).float().permute(2, 0, 1) / 127.5 - 1.0
        
        # --- Condition Input (reference image + target feature map) ---
        condition_tensor = torch.cat([ref_tensor, feature_tensor], dim=0)
        
        return condition_tensor, target_tensor, feature_tensor
    
    def _create_pose_map(self, keypoints: List[List[float]]) -> np.ndarray:
        """Create pose map from keypoints."""
        pose_map = np.zeros((self.input_size, self.input_size, 1), dtype=np.uint8)
        
        for kp in keypoints:
            if len(kp) >= 2:
                x, y = int(kp[0] * self.input_size), int(kp[1] * self.input_size)
                if 0 <= x < self.input_size and 0 <= y < self.input_size:
                    cv2.circle(pose_map, (x, y), 3, (255,), -1)
        
        return pose_map
    
    def _create_feature_map(self, keypoints: List[List[float]]) -> np.ndarray:
        """Create feature map from edge/corner keypoints."""
        feature_map = np.zeros((self.input_size, self.input_size, 1), dtype=np.uint8)
        
        for kp in keypoints:
            if len(kp) >= 2:
                x, y = int(kp[0] * self.input_size), int(kp[1] * self.input_size)
                if 0 <= x < self.input_size and 0 <= y < self.input_size:
                    cv2.circle(feature_map, (x, y), 2, (255,), -1)
        
        return feature_map
    
    def _apply_augmentation(self, image: np.ndarray, 
                          feature_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random horizontal flip
        if config.get_bool('training', 'horizontal_flip', True) and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            feature_map = cv2.flip(feature_map, 1)
        
        # Random rotation
        rotation_range = config.get_float('training', 'rotation_range', 15.0)
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            center = (self.input_size // 2, self.input_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            image = cv2.warpAffine(image, rotation_matrix, (self.input_size, self.input_size))
            feature_map = cv2.warpAffine(feature_map, rotation_matrix, (self.input_size, self.input_size))
        
        # Random brightness adjustment
        brightness_range = config.get_float('training', 'brightness_range', 0.2)
        if brightness_range > 0:
            brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        return image, feature_map


class ModelTrainer:
    """
    Trainer for generative models.
    
    This component handles training of cGAN models for different object categories.
    """
    
    def __init__(self, device: torch.device):
        """Initialize the model trainer."""
        self.device = device
        
        # Training configuration
        self.learning_rate = config.get_float('training', 'learning_rate', 0.0002)
        self.beta1 = config.get_float('training', 'beta1', 0.5)
        self.beta2 = config.get_float('training', 'beta2', 0.999)
        self.num_epochs = config.get_int('training', 'num_epochs', 200)
        self.batch_size = config.get_int('models', 'batch_size', 8)
        self.save_checkpoint_every = config.get_int('training', 'save_checkpoint_every', 10)
        
        # Loss weights
        self.adversarial_weight = config.get_float('training', 'adversarial_loss_weight', 1.0)
        self.reconstruction_weight = config.get_float('training', 'reconstruction_loss_weight', 100.0)
        self.perceptual_weight = config.get_float('training', 'perceptual_loss_weight', 10.0)
        
        # Setup logging
        self.enable_tensorboard = config.get_bool('logging', 'enable_tensorboard', True)
        self.tensorboard_dir = Path(config.get_str('logging', 'tensorboard_log_dir', './logs/tensorboard'))
        
        # Initialize loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        
        # Initialize perceptual loss (if available)
        self.perceptual_loss = None
        try:
            import lpips
            self.perceptual_loss = lpips.LPIPS(net='alex').to(device)
            logging.info("   🧠 Perceptual loss (LPIPS) initialized")
        except ImportError:
            logging.warning("LPIPS not available, perceptual loss disabled")
        
        logging.info("🎓 Model Trainer initialized")
        logging.info(f"   Device: {self.device}")
        logging.info(f"   Learning rate: {self.learning_rate}")
        logging.info(f"   Epochs: {self.num_epochs}")
        logging.info(f"   Batch size: {self.batch_size}")
    
    @track_performance
    def train_human_model(self, human_objects: Dict[str, List[Dict[str, Any]]],
                         config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train human generation model."""
        logging.info("👤 Training human generation model...")
        
        # Create dataset
        dataset = ObjectDataset(human_objects, 'human', 
                              input_size=config.get_int('models', 'human_input_size', 256))
        
        if len(dataset) == 0:
            logging.warning("No valid human training pairs found.")
            return {'error': 'No valid human training data'}
        
        # Initialize model
        model = HumanCGAN(
            input_size=config.get_int('models', 'human_input_size', 256),
            keypoint_channels=config.get_int('models', 'human_keypoint_channels', 17)
        ).to(self.device)
        
        # Train model
        result = self._train_model(model, dataset, 'human', config_override)
        
        # Save trained model
        model_path = Path(config.get_str('models', 'human_model_path', './models/human_cgan.pth'))
        model_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'generator': model.generator.state_dict(),
            'discriminator': model.discriminator.state_dict(),
            'training_config': config_override or {},
            'dataset_size': len(dataset)
        }, model_path)
        
        result['model_path'] = str(model_path)
        logging.info(f"✅ Human model saved: {model_path}")
        
        return result
    
    @track_performance
    def train_animal_model(self, animal_objects: Dict[str, List[Dict[str, Any]]],
                          config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train animal generation model."""
        logging.info("🐾 Training animal generation model...")
        
        # Create dataset
        dataset = ObjectDataset(animal_objects, 'animal',
                              input_size=config.get_int('models', 'animal_input_size', 256))
        
        if len(dataset) == 0:
            logging.warning("No valid animal training pairs found.")
            return {'error': 'No valid animal training data'}
        
        # Initialize model
        model = AnimalCGAN(
            input_size=config.get_int('models', 'animal_input_size', 256),
            keypoint_channels=config.get_int('models', 'animal_keypoint_channels', 20)
        ).to(self.device)
        
        # Train model
        result = self._train_model(model, dataset, 'animal', config_override)
        
        # Save trained model
        model_path = Path(config.get_str('models', 'animal_model_path', './models/animal_cgan.pth'))
        model_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'generator': model.generator.state_dict(),
            'discriminator': model.discriminator.state_dict(),
            'training_config': config_override or {},
            'dataset_size': len(dataset)
        }, model_path)
        
        result['model_path'] = str(model_path)
        logging.info(f"✅ Animal model saved: {model_path}")
        
        return result
    
    @track_performance
    def train_other_model(self, other_objects: Dict[str, List[Dict[str, Any]]],
                         config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train other objects generation model."""
        logging.info("📦 Training other objects generation model...")
        
        # Create dataset
        dataset = ObjectDataset(other_objects, 'other',
                              input_size=config.get_int('models', 'other_input_size', 256))
        
        if len(dataset) == 0:
            logging.warning("No valid other training pairs found.")
            return {'error': 'No valid other objects training data'}
        
        # Initialize model
        model = OtherCGAN(
            input_size=config.get_int('models', 'other_input_size', 256),
            feature_channels=config.get_int('models', 'other_feature_channels', 50)
        ).to(self.device)
        
        # Train model
        result = self._train_model(model, dataset, 'other', config_override)
        
        # Save trained model
        model_path = Path(config.get_str('models', 'other_model_path', './models/other_cgan.pth'))
        model_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'generator': model.generator.state_dict(),
            'discriminator': model.discriminator.state_dict(),
            'training_config': config_override or {},
            'dataset_size': len(dataset)
        }, model_path)
        
        result['model_path'] = str(model_path)
        logging.info(f"✅ Other objects model saved: {model_path}")
        
        return result
    
    def _train_model(self, model: nn.Module, dataset: ObjectDataset, 
                    category: str, config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a generative model."""
        start_time = time.time()
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        # Setup optimizers
        generator_optimizer = optim.Adam(
            model.generator.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2)
        )
        
        discriminator_optimizer = optim.Adam(
            model.discriminator.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta1, self.beta2)
        )
        
        # Setup tensorboard logging
        writer = None
        if self.enable_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = self.tensorboard_dir / f"{category}_{int(time.time())}"
            writer = SummaryWriter(str(log_dir))
            logging.info(f"   📊 Tensorboard logging: {log_dir}")
        
        # Training metrics
        generator_losses = []
        discriminator_losses = []
        
        logging.info(f"🎯 Starting training: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0
            
            for batch_idx, data in enumerate(dataloader):
                if data[0] is None:
                    logging.warning(f"Skipping empty batch at epoch {epoch+1}, batch {batch_idx}")
                    continue
                condition_input, target_image, feature_map = data

                # Move to device
                condition_input = condition_input.to(self.device)
                target_image = target_image.to(self.device)
                feature_map = feature_map.to(self.device)
                
                batch_size = target_image.size(0)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1, 5, 5).to(self.device)  # PatchGAN
                fake_labels = torch.zeros(batch_size, 1, 5, 5).to(self.device)
                
                # Train Discriminator
                discriminator_optimizer.zero_grad()
                
                # Real images
                real_input = torch.cat([target_image, condition_input[:, :3]], dim=1)  # target + reference
                if category == 'other':
                    real_pred_main, real_pred_fine = model.discriminate(target_image, condition_input[:, :3])
                    real_loss = (self.adversarial_loss(real_pred_main, real_labels) + 
                               self.adversarial_loss(real_pred_fine, real_labels[:, :, :real_pred_fine.size(2), :real_pred_fine.size(3)])) / 2
                else:
                    real_pred = model.discriminate(target_image, condition_input[:, :3])
                    real_loss = self.adversarial_loss(real_pred, real_labels)
                
                # Fake images
                fake_images = model.generator(condition_input)
                if category == 'other':
                    fake_pred_main, fake_pred_fine = model.discriminate(fake_images.detach(), condition_input[:, :3])
                    fake_loss = (self.adversarial_loss(fake_pred_main, fake_labels) + 
                               self.adversarial_loss(fake_pred_fine, fake_labels[:, :, :fake_pred_fine.size(2), :fake_pred_fine.size(3)])) / 2
                else:
                    fake_pred = model.discriminate(fake_images.detach(), condition_input[:, :3])
                    fake_loss = self.adversarial_loss(fake_pred, fake_labels)
                
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                discriminator_optimizer.step()
                
                # Train Generator
                generator_optimizer.zero_grad()
                
                # Adversarial loss
                if category == 'other':
                    fake_pred_main, fake_pred_fine = model.discriminate(fake_images, condition_input[:, :3])
                    adv_loss = (self.adversarial_loss(fake_pred_main, real_labels) + 
                              self.adversarial_loss(fake_pred_fine, real_labels[:, :, :fake_pred_fine.size(2), :fake_pred_fine.size(3)])) / 2
                else:
                    fake_pred = model.discriminate(fake_images, condition_input[:, :3])
                    adv_loss = self.adversarial_loss(fake_pred, real_labels)
                
                # Reconstruction loss
                recon_loss = self.reconstruction_loss(fake_images, target_image)
                
                # Perceptual loss (if available)
                perceptual_loss_value = 0
                if self.perceptual_loss is not None:
                    perceptual_loss_value = self.perceptual_loss(fake_images, target_image).mean()
                
                # Total generator loss
                gen_loss = (self.adversarial_weight * adv_loss + 
                           self.reconstruction_weight * recon_loss + 
                           self.perceptual_weight * perceptual_loss_value)
                
                gen_loss.backward()
                generator_optimizer.step()
                
                # Accumulate losses
                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
                
                # Log to tensorboard
                if writer and batch_idx % 50 == 0:
                    global_step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar(f'Loss/Generator', gen_loss.item(), global_step)
                    writer.add_scalar(f'Loss/Discriminator', disc_loss.item(), global_step)
                    writer.add_scalar(f'Loss/Reconstruction', recon_loss.item(), global_step)
                    if perceptual_loss_value > 0:
                        writer.add_scalar(f'Loss/Perceptual', perceptual_loss_value, global_step)
            
            # Epoch summary
            avg_gen_loss = epoch_gen_loss / len(dataloader)
            avg_disc_loss = epoch_disc_loss / len(dataloader)
            
            generator_losses.append(avg_gen_loss)
            discriminator_losses.append(avg_disc_loss)
            
            epoch_time = time.time() - epoch_start
            
            logging.info(f"   Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s): "
                        f"Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_checkpoint_every == 0:
                checkpoint_path = self.tensorboard_dir / f"{category}_checkpoint_epoch_{epoch+1}.pth"
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'generator': model.generator.state_dict(),
                    'discriminator': model.discriminator.state_dict(),
                    'generator_optimizer': generator_optimizer.state_dict(),
                    'discriminator_optimizer': discriminator_optimizer.state_dict(),
                    'generator_losses': generator_losses,
                    'discriminator_losses': discriminator_losses
                }, checkpoint_path)
        
        # Close tensorboard writer
        if writer:
            writer.close()
        
        training_time = time.time() - start_time
        
        result = {
            'category': category,
            'training_time': training_time,
            'epochs_completed': self.num_epochs,
            'dataset_size': len(dataset),
            'final_generator_loss': generator_losses[-1] if generator_losses else 0,
            'final_discriminator_loss': discriminator_losses[-1] if discriminator_losses else 0,
            'generator_losses': generator_losses,
            'discriminator_losses': discriminator_losses
        }
        
        logging.info(f"✅ {category.capitalize()} model training completed in {training_time:.1f}s")
        
        return result
