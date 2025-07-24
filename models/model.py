import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import os
from pathlib import Path

# Import our custom models
try:
    from models.generator import PokemonGenerator
    from models.discriminator import PokemonDiscriminator, PatchGANDiscriminator
except ImportError:
    from generator import PokemonGenerator
    from discriminator import PokemonDiscriminator, PatchGANDiscriminator

class PokemonGAN(nn.Module):
    """
    Complete Pokemon Card GAN with conditional generation
    
    Features:
    - Conditional generation based on Pokemon type and stats
    - Auxiliary losses for better training stability
    - Support for both regular and PatchGAN discriminators
    - Progressive training capabilities
    """
    
    def __init__(self,
                 latent_dim: int = 128,
                 num_types: int = 11,
                 stats_dim: int = 4,
                 generator_lr: float = 0.0002,
                 discriminator_lr: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 use_patch_gan: bool = False,
                 device: str = 'cuda'):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_types = num_types
        self.stats_dim = stats_dim
        self.device = device
        self.use_patch_gan = use_patch_gan
        
        # Initialize models
        self.generator = PokemonGenerator(
            latent_dim=latent_dim,
            num_types=num_types,
            stats_dim=stats_dim
        ).to(device)
        
        if use_patch_gan:
            self.discriminator = PatchGANDiscriminator(
                num_types=num_types,
                stats_dim=stats_dim
            ).to(device)
        else:
            self.discriminator = PokemonDiscriminator(
                num_types=num_types,
                stats_dim=stats_dim
            ).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=generator_lr, 
            betas=(beta1, beta2)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()  # For type prediction
        self.regression_loss = nn.MSELoss()  # For stats prediction
        
        # Loss weights
        self.lambda_aux_type = 1.0    # Type auxiliary loss weight
        self.lambda_aux_stats = 0.5   # Stats auxiliary loss weight
        
        # Training statistics
        self.training_stats = {
            'g_loss': [],
            'd_loss': [],
            'g_aux_loss': [],
            'd_aux_loss': []
        }
    
    def generate_noise(self, batch_size: int) -> torch.Tensor:
        """Generate random noise for the generator"""
        return torch.randn(batch_size, self.latent_dim, device=self.device)
    
    def train_discriminator(self, real_images: torch.Tensor, real_types: torch.Tensor,
                          real_stats: torch.Tensor) -> Dict[str, float]:
        """
        Train discriminator on real and fake images
        
        Args:
            real_images: Real Pokemon artwork (batch_size, 3, 256, 256)
            real_types: Real type labels (batch_size, num_types)
            real_stats: Real stats (batch_size, stats_dim)
            
        Returns:
            Dictionary of discriminator losses
        """
        batch_size = real_images.size(0)
        self.d_optimizer.zero_grad()
        
        # Labels for real and fake
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # === Train on real images ===
        if self.use_patch_gan:
            real_patch_logits, real_pred_types, real_pred_stats = self.discriminator(real_images)
            # For PatchGAN, we need to reshape labels to match patch output
            patch_size = real_patch_logits.shape[-1]
            real_patch_labels = real_labels.expand(-1, patch_size * patch_size).view(batch_size, 1, patch_size, patch_size)
            d_real_loss = self.adversarial_loss(real_patch_logits, real_patch_labels)
        else:
            real_logits, real_pred_types, real_pred_stats = self.discriminator(real_images, real_types, real_stats)
            d_real_loss = self.adversarial_loss(real_logits, real_labels)
        
        # Auxiliary losses on real images
        real_type_labels = torch.argmax(real_types, dim=1)  # Convert one-hot to class indices
        d_aux_type_loss = self.auxiliary_loss(real_pred_types, real_type_labels)
        d_aux_stats_loss = self.regression_loss(real_pred_stats, real_stats)
        
        # === Train on fake images ===
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise, real_types, real_stats).detach()  # Detach to avoid training G
        
        if self.use_patch_gan:
            fake_patch_logits, _, _ = self.discriminator(fake_images)
            fake_patch_labels = fake_labels.expand(-1, patch_size * patch_size).view(batch_size, 1, patch_size, patch_size)
            d_fake_loss = self.adversarial_loss(fake_patch_logits, fake_patch_labels)
        else:
            fake_logits = self.discriminator.discriminate_only(fake_images)
            d_fake_loss = self.adversarial_loss(fake_logits, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + \
                self.lambda_aux_type * d_aux_type_loss + \
                self.lambda_aux_stats * d_aux_stats_loss
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_real_loss': d_real_loss.item(),
            'd_fake_loss': d_fake_loss.item(),
            'd_aux_type_loss': d_aux_type_loss.item(),
            'd_aux_stats_loss': d_aux_stats_loss.item()
        }
    
    def train_generator(self, batch_size: int, real_types: torch.Tensor,
                       real_stats: torch.Tensor) -> Dict[str, float]:
        """
        Train generator to fool discriminator
        
        Args:
            batch_size: Size of the batch
            real_types: Type conditions (batch_size, num_types)
            real_stats: Stats conditions (batch_size, stats_dim)
            
        Returns:
            Dictionary of generator losses
        """
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = self.generate_noise(batch_size)
        fake_images = self.generator(noise, real_types, real_stats)
        
        # Labels (we want discriminator to think these are real)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        # Get discriminator predictions on fake images
        if self.use_patch_gan:
            fake_patch_logits, fake_pred_types, fake_pred_stats = self.discriminator(fake_images)
            patch_size = fake_patch_logits.shape[-1]
            real_patch_labels = real_labels.expand(-1, patch_size * patch_size).view(batch_size, 1, patch_size, patch_size)
            g_adv_loss = self.adversarial_loss(fake_patch_logits, real_patch_labels)
        else:
            fake_logits, fake_pred_types, fake_pred_stats = self.discriminator(fake_images, real_types, real_stats)
            g_adv_loss = self.adversarial_loss(fake_logits, real_labels)
        
        # Auxiliary losses - generator should produce images that match the conditions
        real_type_labels = torch.argmax(real_types, dim=1)
        g_aux_type_loss = self.auxiliary_loss(fake_pred_types, real_type_labels)
        g_aux_stats_loss = self.regression_loss(fake_pred_stats, real_stats)
        
        # Total generator loss
        g_loss = g_adv_loss + \
                self.lambda_aux_type * g_aux_type_loss + \
                self.lambda_aux_stats * g_aux_stats_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_aux_type_loss': g_aux_type_loss.item(),
            'g_aux_stats_loss': g_aux_stats_loss.item()
        }
    
    def train_step(self, real_images: torch.Tensor, real_types: torch.Tensor,
                  real_stats: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            real_images: Real Pokemon artwork (batch_size, 3, 256, 256)
            real_types: Real type labels (batch_size, num_types)
            real_stats: Real stats (batch_size, stats_dim)
            
        Returns:
            Dictionary of all losses
        """
        batch_size = real_images.size(0)
        
        # Train discriminator
        d_losses = self.train_discriminator(real_images, real_types, real_stats)
        
        # Train generator
        g_losses = self.train_generator(batch_size, real_types, real_stats)
        
        # Combine losses
        losses = {**d_losses, **g_losses}
        
        # Update training statistics
        self.training_stats['g_loss'].append(g_losses['g_loss'])
        self.training_stats['d_loss'].append(d_losses['d_loss'])
        
        return losses
    
    def generate_samples(self, types: torch.Tensor, stats: torch.Tensor,
                        num_samples: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples with given conditions
        
        Args:
            types: Type conditions (batch_size, num_types)
            stats: Stats conditions (batch_size, stats_dim)
            num_samples: Override batch size if provided
            
        Returns:
            Generated Pokemon artwork (batch_size, 3, 256, 256)
        """
        if num_samples is not None:
            batch_size = num_samples
            types = types[:batch_size]
            stats = stats[:batch_size]
        else:
            batch_size = types.size(0)
        
        return self.generator.generate_samples(batch_size, types, stats, self.device)
    
    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': {
                'latent_dim': self.latent_dim,
                'num_types': self.num_types,
                'stats_dim': self.stats_dim,
                'use_patch_gan': self.use_patch_gan
            }
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if load_optimizer:
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint.get('epoch', 0)

def test_pokemon_gan():
    """Test the complete Pokemon GAN"""
    print("Testing Complete Pokemon GAN...")
    
    # Create GAN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    gan = PokemonGAN(
        latent_dim=128,
        num_types=11,
        stats_dim=4,
        use_patch_gan=False,
        device=device
    )
    
    # Test data
    batch_size = 4
    real_images = torch.randn(batch_size, 3, 256, 256, device=device)
    real_types = torch.zeros(batch_size, 11, device=device)
    real_types[:, 0] = 1  # Fire type
    real_stats = torch.randn(batch_size, 4, device=device)
    
    # Test training step
    print("Testing training step...")
    losses = gan.train_step(real_images, real_types, real_stats)
    
    print("Training losses:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    # Test sample generation
    print("\nTesting sample generation...")
    samples = gan.generate_samples(real_types, real_stats)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    checkpoint_path = "test_checkpoint.pth"
    gan.save_checkpoint(checkpoint_path, epoch=1)
    
    # Create new GAN and load checkpoint
    gan2 = PokemonGAN(device=device)
    loaded_epoch = gan2.load_checkpoint(checkpoint_path)
    print(f"Loaded epoch: {loaded_epoch}")
    
    # Cleanup
    os.remove(checkpoint_path)
    
    # Test with PatchGAN
    print("\nTesting with PatchGAN discriminator...")
    patch_gan = PokemonGAN(use_patch_gan=True, device=device)
    patch_losses = patch_gan.train_step(real_images, real_types, real_stats)
    
    print("PatchGAN losses:")
    for loss_name, loss_value in patch_losses.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    print("\nPokemon GAN test completed successfully!")

if __name__ == "__main__":
    test_pokemon_gan()