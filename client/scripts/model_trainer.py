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


class ObjectDataset(Dataset):
    """Dataset for training object generation models on vector-based inputs."""
    
    def __init__(self, objects_data: List[Dict[str, Any]], category: str, 
                 input_size: int = 256, augment: bool = False): # Augmentation disabled for now
        """
        Initialize object dataset.
        
        Args:
            objects_data: List of object data from the server.
            category: Object category (human, animal, other).
            input_size: The target output image size.
            augment: Whether to apply data augmentation (currently not supported for vectors).
        """
        self.category = category
        self.input_size = input_size
        self.augment = augment
        
        # Filter for valid objects that have all the necessary data
        self.valid_objects = []
        for obj in objects_data:
            if self._validate_object(obj):
                self.valid_objects.append(obj)
        
        logging.info(f"📚 {category.capitalize()} dataset: {len(self.valid_objects)}/{len(objects_data)} valid objects")
    
    def _validate_object(self, obj: Dict[str, Any]) -> bool:
        """Validate that an object has the required data for training."""
        if 'cropped_image' not in obj or obj['cropped_image'] is None:
            return False
        if 'v_appearance' not in obj or obj['v_appearance'] is None:
            return False
        if 'p_pose' not in obj or not obj['p_pose'].get('points'):
            return False
        return True
    
    def __len__(self) -> int:
        return len(self.valid_objects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training item.
        
        Returns:
            Tuple of (v_appearance, p_t, target_image)
        """
        obj = self.valid_objects[idx]
        
        # Load and process the target image (ground truth)
        target_image = obj['cropped_image']
        if isinstance(target_image, str):
            target_image = cv2.imread(target_image)
        target_image = cv2.resize(target_image, (self.input_size, self.input_size))
        target_tensor = torch.from_numpy(target_image).float().permute(2, 0, 1) / 127.5 - 1.0
        
        # Get the appearance vector
        v_appearance = np.array(obj['v_appearance'])
        v_appearance_tensor = torch.from_numpy(v_appearance).float()
        
        # Get the pose vector
        p_pose_data = obj['p_pose'].get('points', [])
        if self.category in ['human', 'animal']:
            p_t = [coord for kp in p_pose_data for coord in kp]
        else:
            p_t = p_pose_data
        p_t_tensor = torch.tensor(p_t, dtype=torch.float32)
        
        return v_appearance_tensor, p_t_tensor, target_tensor


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
    def train_human_model(self, human_objects: List[Dict[str, Any]], 
                         config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train human generation model."""
        logging.info("👤 Training human generation model...")
        
        # Create dataset
        dataset = ObjectDataset(human_objects, 'human', 
                              input_size=config.get_int('models', 'human_input_size', 256))
        
        if len(dataset) == 0:
            return {'error': 'No valid human training data'}
        
        # Initialize model
        human_vector_size = 2048 + config.get_int('models', 'human_keypoint_channels', 17) * 3
        model = HumanCGAN(
            input_size=config.get_int('models', 'human_input_size', 256),
            vector_input_size=human_vector_size
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
    def train_animal_model(self, animal_objects: List[Dict[str, Any]], 
                          config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train animal generation model."""
        logging.info("🐾 Training animal generation model...")
        
        # Create dataset
        dataset = ObjectDataset(animal_objects, 'animal',
                              input_size=config.get_int('models', 'animal_input_size', 256))
        
        if len(dataset) == 0:
            return {'error': 'No valid animal training data'}
        
        # Initialize model
        animal_vector_size = 2048 + config.get_int('models', 'animal_keypoint_channels', 20) * 3
        model = AnimalCGAN(
            input_size=config.get_int('models', 'animal_input_size', 256),
            vector_input_size=animal_vector_size
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
    def train_other_model(self, other_objects: List[Dict[str, Any]], 
                         config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train other objects generation model."""
        logging.info("📦 Training other objects generation model...")
        
        # Create dataset
        dataset = ObjectDataset(other_objects, 'other',
                              input_size=config.get_int('models', 'other_input_size', 256))
        
        if len(dataset) == 0:
            return {'error': 'No valid other objects training data'}
        
        # Initialize model
        other_vector_size = 2048 + 4
        model = OtherCGAN(
            input_size=config.get_int('models', 'other_input_size', 256),
            vector_input_size=other_vector_size
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
        """Train a generative model using the new vector-based inputs."""
        start_time = time.time()
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)

        generator_optimizer = optim.Adam(model.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

        writer = None
        if self.enable_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = self.tensorboard_dir / f"{category}_{int(time.time())}"
            writer = SummaryWriter(str(log_dir))
            logging.info(f"   📊 Tensorboard logging: {log_dir}")
        
        generator_losses, discriminator_losses = [], []
        logging.info(f"🎯 Starting training: {len(dataset)} samples, {len(dataloader)} batches")

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            epoch_gen_loss, epoch_disc_loss = 0.0, 0.0
            
            for batch_idx, (v_appearance, p_t, target_image) in enumerate(dataloader):
                v_appearance = v_appearance.to(self.device)
                p_t = p_t.to(self.device)
                target_image = target_image.to(self.device)
                batch_size = target_image.size(0)

                # --- Train Discriminator ---
                discriminator_optimizer.zero_grad()
                
                # Real images
                real_pred = model.discriminate(target_image, p_t)
                real_labels = torch.ones_like(real_pred).to(self.device)
                real_loss = self.adversarial_loss(real_pred, real_labels)
                
                # Fake images
                input_vec = torch.cat([v_appearance, p_t], dim=1)
                fake_images = model.generator(input_vec)
                fake_pred = model.discriminate(fake_images.detach(), p_t)
                fake_labels = torch.zeros_like(fake_pred).to(self.device)
                fake_loss = self.adversarial_loss(fake_pred, fake_labels)
                
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                discriminator_optimizer.step()
                
                # --- Train Generator ---
                generator_optimizer.zero_grad()
                
                # Adversarial loss
                fake_pred = model.discriminate(fake_images, p_t)
                adv_loss = self.adversarial_loss(fake_pred, real_labels)
                
                # Reconstruction loss (L1)
                recon_loss = self.reconstruction_loss(fake_images, target_image)
                
                # Perceptual loss
                perceptual_loss_value = 0
                if self.perceptual_loss is not None:
                    perceptual_loss_value = self.perceptual_loss(fake_images, target_image).mean()
                
                gen_loss = (self.adversarial_weight * adv_loss + 
                           self.reconstruction_weight * recon_loss + 
                           self.perceptual_weight * perceptual_loss_value)
                gen_loss.backward()
                generator_optimizer.step()
                
                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
                
                if writer and batch_idx % 50 == 0:
                    global_step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar(f'Loss/Generator', gen_loss.item(), global_step)
                    writer.add_scalar(f'Loss/Discriminator', disc_loss.item(), global_step)
            
            avg_gen_loss = epoch_gen_loss / len(dataloader)
            avg_disc_loss = epoch_disc_loss / len(dataloader)
            generator_losses.append(avg_gen_loss)
            discriminator_losses.append(avg_disc_loss)
            logging.info(f"   Epoch {epoch+1}/{self.num_epochs} ({(time.time() - epoch_start):.1f}s): Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            if (epoch + 1) % self.save_checkpoint_every == 0:
                # Save checkpoint logic...
                pass # Simplified for brevity

        if writer:
            writer.close()

        return {
            'category': category,
            'training_time': time.time() - start_time,
            'final_generator_loss': generator_losses[-1] if generator_losses else 0
        }
