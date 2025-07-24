#!/usr/bin/env python3
"""
Pokemon GAN Training Script
Run from project root: python train_pokemon_gan.py
"""

import torch
import torch.nn as nn
import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Fix matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import our modules (absolute imports from root)
from data.dataset import PokemonDataModule
from models.model import PokemonGAN
from models.utils import (
    save_sample_grid, 
    create_pokemon_type_samples,
    log_model_info,
    count_parameters
)

def setup_training_directories():
    """Create necessary directories for training outputs"""
    directories = [
        'outputs/checkpoints',
        'outputs/generated_samples',
        'outputs/logs',
        'outputs/training_progress'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Training directories created")

def load_data_info(data_dir='processed_data'):
    """Load dataset information"""
    info_path = os.path.join(data_dir, 'dataset_info.json')
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    
    with open(info_path, 'r') as f:
        data_info = json.load(f)
    
    print(f"✓ Dataset info loaded: {data_info['num_types']} types, {data_info['image_size']}x{data_info['image_size']} images")
    
    return data_info

def create_fixed_samples(data_info, device, num_samples=16):
    """Create fixed samples for consistent progress tracking"""
    type_mappings = data_info['type_to_idx']
    
    # Create samples covering all Pokemon types
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

def save_training_progress(losses, epoch, save_dir='outputs/logs'):
    """Save training progress plots"""
    if len(losses['g_loss']) == 0:
        return
        
    epochs = range(1, len(losses['g_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generator and Discriminator losses
    ax1.plot(epochs, losses['g_loss'], label='Generator', color='blue')
    ax1.plot(epochs, losses['d_loss'], label='Discriminator', color='red')
    ax1.set_title('Adversarial Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Auxiliary losses (only if available and same length)
    if ('g_aux_loss' in losses and 'd_aux_loss' in losses and 
        len(losses['g_aux_loss']) == len(epochs) and len(losses['d_aux_loss']) == len(epochs)):
        ax2.plot(epochs, losses['g_aux_loss'], label='Generator Aux', color='lightblue')
        ax2.plot(epochs, losses['d_aux_loss'], label='Discriminator Aux', color='lightcoral')
        ax2.set_title('Auxiliary Losses')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Auxiliary losses\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Auxiliary Losses')
    
    # Loss ratio
    if len(losses['g_loss']) > 0 and len(losses['d_loss']) > 0:
        ratios = [g/d if d > 0 else 0 for g, d in zip(losses['g_loss'], losses['d_loss'])]
        ax3.plot(epochs, ratios, color='green')
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Generator/Discriminator Loss Ratio')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Ratio')
        ax3.grid(True)
    
    # Recent losses (last 50% of training)
    if len(epochs) > 4:
        recent_start = max(1, int(0.5 * len(epochs)))
        recent_epochs = list(epochs)[recent_start-1:]
        recent_g = losses['g_loss'][recent_start-1:]
        recent_d = losses['d_loss'][recent_start-1:]
        
        ax4.plot(recent_epochs, recent_g, label='Generator', color='blue')
        ax4.plot(recent_epochs, recent_d, label='Discriminator', color='red')
        ax4.set_title('Recent Losses (Last 50%)')
    else:
        ax4.plot(epochs, losses['g_loss'], label='Generator', color='blue')
        ax4.plot(epochs, losses['d_loss'], label='Discriminator', color='red')
        ax4.set_title('All Losses')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_progress_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_pokemon_gan(
    data_dir='processed_data',
    num_epochs=100,
    batch_size=32,
    latent_dim=128,
    generator_lr=0.0002,
    discriminator_lr=None,  # If None, use same as generator_lr
    save_frequency=10,
    sample_frequency=5,
    use_patch_gan=False,
    resume_from=None,
    device='auto',
    checkpoint_dir='outputs/checkpoints',
    sample_dir='outputs/generated_samples', 
    log_dir='outputs/logs'
):
    """
    Main training function for Pokemon GAN
    
    Args:
        data_dir: Directory containing processed data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        latent_dim: Latent dimension for generator
        learning_rate: Learning rate for optimizers
        save_frequency: Save checkpoint every N epochs
        sample_frequency: Generate samples every N epochs
        use_patch_gan: Whether to use PatchGAN discriminator
        resume_from: Path to checkpoint to resume from
        device: Device to train on ('auto', 'cuda', 'cpu')
    """
    
    # Setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use same LR for both if discriminator_lr not specified
    if discriminator_lr is None:
        discriminator_lr = generator_lr
    
    print(f"🚀 Starting Pokemon GAN Training")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
    print(f"Generator LR: {generator_lr}, Discriminator LR: {discriminator_lr}")
    print(f"Using {'PatchGAN' if use_patch_gan else 'Standard'} Discriminator")
    
    # Create custom output directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_info = load_data_info(data_dir)
    
    data_module = PokemonDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"✓ Data loaded: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    training_losses = {'g_loss': [], 'd_loss': [], 'g_aux_loss': [], 'd_aux_loss': []}
    
    if resume_from and os.path.exists(resume_from):
        print(f"📂 Resuming from checkpoint: {resume_from}")
        start_epoch = gan.load_checkpoint(resume_from)
        training_losses = gan.training_stats
        print(f"✓ Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n🎯 Starting training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch + 1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        gan.generator.train()
        gan.discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_g_aux_losses = []
        epoch_d_aux_losses = []
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        
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
            epoch_g_aux_losses.append(losses.get('g_aux_type_loss', 0) + losses.get('g_aux_stats_loss', 0))
            epoch_d_aux_losses.append(losses.get('d_aux_type_loss', 0) + losses.get('d_aux_stats_loss', 0))
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{losses['g_loss']:.3f}",
                'D_loss': f"{losses['d_loss']:.3f}"
            })
        
        # Calculate epoch averages
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_g_aux_loss = sum(epoch_g_aux_losses) / len(epoch_g_aux_losses)
        avg_d_aux_loss = sum(epoch_d_aux_losses) / len(epoch_d_aux_losses)
        
        # Store epoch losses
        training_losses['g_loss'].append(avg_g_loss)
        training_losses['d_loss'].append(avg_d_loss)
        training_losses['g_aux_loss'].append(avg_g_aux_loss)
        training_losses['d_aux_loss'].append(avg_d_aux_loss)
        
        # Update GAN training stats
        gan.training_stats = training_losses
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch} | G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Generate samples
        if epoch % sample_frequency == 0 or epoch == 1:
            print(f"📸 Generating samples...")
            gan.generator.eval()
            
            with torch.no_grad():
                fixed_samples = gan.generate_samples(fixed_types, fixed_stats)
                
                save_sample_grid(
                    fixed_samples,
                    f'outputs/generated_samples/epoch_{epoch:03d}.png',
                    nrow=4,
                    title=f'Epoch {epoch} - Pokemon Samples'
                )
        
        # Save checkpoint
        if epoch % save_frequency == 0 or epoch == num_epochs:
            checkpoint_path = f'outputs/checkpoints/pokemon_gan_epoch_{epoch:03d}.pth'
            gan.save_checkpoint(checkpoint_path, epoch, {
                'training_losses': training_losses,
                'data_info': data_info
            })
        
        # Save training progress
        if epoch % (save_frequency * 2) == 0 or epoch == num_epochs:
            save_training_progress(training_losses, epoch)
    
    print(f"\n🎉 Training completed!")
    print(f"Final checkpoint: outputs/checkpoints/pokemon_gan_epoch_{num_epochs:03d}.pth")
    print(f"Generated samples: outputs/generated_samples/")
    print(f"Training logs: outputs/logs/")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train Pokemon GAN')
    
    parser.add_argument('--data_dir', type=str, default='processed_pokemon_data',
                       help='Directory containing processed data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension for generator')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--sample_freq', type=int, default=5,
                       help='Generate samples every N epochs')
    parser.add_argument('--patch_gan', action='store_true',
                       help='Use PatchGAN discriminator')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Run preprocessing first: python data/preprocessor.py")
        return
    
    # Start training
    train_pokemon_gan(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        generator_lr=args.lr,
        save_frequency=args.save_freq,
        sample_frequency=args.sample_freq,
        use_patch_gan=args.patch_gan,
        resume_from=args.resume,
        device=args.device
    )

if __name__ == "__main__":
    main()