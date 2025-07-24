#!/usr/bin/env python3
"""
Pokemon GAN Training Script - Simplified with Dashboard Logger
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from tqdm import tqdm

from dashboard_logger import initialize_training

# Import your model-specific modules
from data.dataset import PokemonDataModule
from models.model import PokemonGAN
from models.utils import (
    save_sample_grid, 
    create_pokemon_type_samples,
    log_model_info
)

def load_data_info(data_dir):
    """Load dataset information"""
    import json
    info_path = os.path.join(data_dir, 'dataset_info.json')
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    
    with open(info_path, 'r') as f:
        data_info = json.load(f)
    
    print(f"DATA_LOADED | Types: {data_info['num_types']} | Image size: {data_info['image_size']}x{data_info['image_size']}")
    return data_info

def create_fixed_samples(data_info, device, num_samples=16):
    """Create fixed samples for consistent progress tracking"""
    type_mappings = data_info['type_to_idx']
    
    type_tensor, stats_tensor, type_names = create_pokemon_type_samples(
        type_mappings, 
        num_samples_per_type=max(1, num_samples // len(type_mappings)),
        device=device
    )
    
    # Trim to exact number we want
    type_tensor = type_tensor[:num_samples]
    stats_tensor = stats_tensor[:num_samples]
    type_names = type_names[:num_samples]
    
    return type_tensor, stats_tensor, type_names

def train_pokemon_gan():
    """Main training function"""
    
    # Initialize training with dashboard logger
    # This handles argument parsing, config loading, directory setup, etc.
    config, run_id, output_dir, logger, dirs, timer = initialize_training(
        required_config_keys=['data_dir', 'num_epochs', 'batch_size']
    )
    
    # Extract config parameters with defaults
    data_dir = config['data_dir']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    latent_dim = config.get('latent_dim', 128)
    generator_lr = config.get('generator_lr', 0.0002)
    discriminator_lr = config.get('discriminator_lr', generator_lr)
    save_frequency = config.get('save_frequency', 10)
    sample_frequency = config.get('sample_frequency', 5)
    use_patch_gan = config.get('use_patch_gan', False)
    device = config.get('device', 'auto')
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Log training metadata
    logger.log_metadata(
        config=config,
        device=device,
        model_type="Pokemon_GAN",
        total_epochs=num_epochs
    )
    
    print(f"DEVICE | Using: {device}")
    print(f"HYPERPARAMS | Epochs: {num_epochs} | Batch: {batch_size} | G_LR: {generator_lr} | D_LR: {discriminator_lr}")
    
    # Load data
    data_info = load_data_info(data_dir)
    
    data_module = PokemonDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"DATA_STATS | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Create GAN
    gan = PokemonGAN(
        latent_dim=latent_dim,
        num_types=data_info['num_types'],
        stats_dim=data_info.get('stats_dim', 4),
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        use_patch_gan=use_patch_gan,
        device=device
    )
    
    # Log model info
    log_model_info(gan.generator, gan.discriminator)
    
    # Create fixed samples for progress tracking
    fixed_types, fixed_stats, fixed_type_names = create_fixed_samples(data_info, device, num_samples=16)
    
    # Training state
    training_losses = {'generator_loss': [], 'discriminator_loss': []}
    
    # Start training timer
    timer.start_training()
    
    print(f"TRAINING_LOOP_START | Starting training for {num_epochs} epochs")
    
    # Main training loop
    for epoch in range(1, num_epochs + 1):
        timer.start_epoch(epoch)
        
        # Training
        gan.generator.train()
        gan.discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        # Training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            real_images = batch['image'].to(device)
            real_types = batch['types'].to(device)
            real_stats = batch['stats'].to(device)
            
            # Train step
            losses = gan.train_step(real_images, real_types, real_stats)
            
            # Collect losses
            epoch_g_losses.append(losses['g_loss'])
            epoch_d_losses.append(losses['d_loss'])
            
            # Log training step (using dashboard logger)
            logger.log_training_step(
                epoch=epoch,
                step=batch_idx,
                generator_loss=losses['g_loss'],
                discriminator_loss=losses['d_loss']
            )
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{losses['g_loss']:.3f}",
                'D_loss': f"{losses['d_loss']:.3f}"
            })
        
        # Calculate epoch averages
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        
        # Store epoch losses for plotting
        training_losses['generator_loss'].append(avg_g_loss)
        training_losses['discriminator_loss'].append(avg_d_loss)
        
        # End epoch timer
        epoch_time = timer.end_epoch(epoch)
        
        # Log epoch metrics (using dashboard logger)
        logger.log_epoch(
            epoch=epoch,
            avg_generator_loss=avg_g_loss,
            avg_discriminator_loss=avg_d_loss,
            epoch_time=epoch_time,
            loss_ratio=avg_g_loss / avg_d_loss if avg_d_loss > 0 else 0
        )
        
        # Generate samples
        if epoch % sample_frequency == 0 or epoch == 1:
            print(f"SAMPLE_GENERATION | Generating samples for epoch {epoch}")
            gan.generator.eval()
            
            with torch.no_grad():
                fixed_samples = gan.generate_samples(fixed_types, fixed_stats)
                
                save_sample_grid(
                    fixed_samples,
                    dirs['samples'] / f'epoch_{epoch:03d}.png',
                    nrow=4,
                    title=f'Epoch {epoch} - Pokemon Samples'
                )
                
                # Also save as latest for dashboard
                save_sample_grid(
                    fixed_samples,  
                    dirs['samples'] / 'latest_samples.png',
                    nrow=4,
                    title=f'Latest Samples (Epoch {epoch})'
                )
        
        # Save checkpoint
        if epoch % save_frequency == 0 or epoch == num_epochs:
            checkpoint_path = dirs['checkpoints'] / f'pokemon_gan_epoch_{epoch:03d}.pth'
            gan.save_checkpoint(str(checkpoint_path), epoch, {
                'training_losses': training_losses,
                'data_info': data_info,
                'run_id': run_id
            })
            logger.log_checkpoint(checkpoint_path, epoch)
        
        # Save training plots using dashboard logger
        if epoch % max(1, save_frequency // 2) == 0 or epoch == num_epochs:
            logger.save_loss_plot(training_losses, epoch)
    
    # Save final model
    final_model_path = dirs['model'] / 'final_model.pth'
    gan.save_checkpoint(str(final_model_path), num_epochs, {
        'training_losses': training_losses,
        'data_info': data_info,
        'run_id': run_id,
        'final_model': True
    })
    
    # End training
    timer.end_training()
    logger.log_completion()
    
    return 0  # Success

def main():
    """Main entry point"""
    try:
        return train_pokemon_gan()
    except Exception as e:
        print(f"ERROR | Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())