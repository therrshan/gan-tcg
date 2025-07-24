import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pathlib import Path

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def initialize_weights(model: nn.Module, init_type: str = 'xavier'):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model to initialize
        init_type: Type of initialization ('xavier', 'kaiming', 'normal')
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)
    print(f"Model initialized with {init_type} initialization")

def create_type_condition_tensor(type_names: List[str], type_mappings: Dict[str, int], 
                                device: str = 'cuda') -> torch.Tensor:
    """
    Create one-hot type condition tensor from type names
    
    Args:
        type_names: List of Pokemon type names
        type_mappings: Dictionary mapping type names to indices
        device: Device to create tensor on
        
    Returns:
        One-hot encoded type tensor (len(type_names), num_types)
    """
    num_types = len(type_mappings)
    batch_size = len(type_names)
    
    type_tensor = torch.zeros(batch_size, num_types, device=device)
    
    for i, type_name in enumerate(type_names):
        if type_name in type_mappings:
            type_tensor[i, type_mappings[type_name]] = 1.0
        else:
            # Default to first type if not found
            type_tensor[i, 0] = 1.0
            print(f"Warning: Type '{type_name}' not found, using default")
    
    return type_tensor

def create_random_stats_tensor(batch_size: int, stats_ranges: Optional[Dict] = None,
                              device: str = 'cuda') -> torch.Tensor:
    """
    Create random stats tensor within realistic ranges
    
    Args:
        batch_size: Number of samples
        stats_ranges: Optional dictionary with ranges for each stat
        device: Device to create tensor on
        
    Returns:
        Random stats tensor (batch_size, 4)
    """
    if stats_ranges is None:
        # Default normalized ranges
        stats_ranges = {
            'hp': (-1.0, 2.0),      # Normalized HP
            'attacks': (0.0, 1.0),   # Number of attacks (normalized)
            'abilities': (0.0, 1.0), # Number of abilities (normalized) 
            'rarity': (0.0, 1.0)     # Rarity (normalized)
        }
    
    stats_tensor = torch.zeros(batch_size, 4, device=device)
    
    # Generate random values within ranges using torch.rand
    hp_min, hp_max = stats_ranges['hp']
    stats_tensor[:, 0] = torch.rand(batch_size, device=device) * (hp_max - hp_min) + hp_min
    
    attacks_min, attacks_max = stats_ranges['attacks']
    stats_tensor[:, 1] = torch.rand(batch_size, device=device) * (attacks_max - attacks_min) + attacks_min
    
    abilities_min, abilities_max = stats_ranges['abilities']
    stats_tensor[:, 2] = torch.rand(batch_size, device=device) * (abilities_max - abilities_min) + abilities_min
    
    rarity_min, rarity_max = stats_ranges['rarity']
    stats_tensor[:, 3] = torch.rand(batch_size, device=device) * (rarity_max - rarity_min) + rarity_min
    
    return stats_tensor

def save_sample_grid(samples: torch.Tensor, filepath: str, nrow: int = 8, 
                    normalize: bool = True, title: Optional[str] = None):
    """
    Save a grid of generated samples
    
    Args:
        samples: Generated samples (batch_size, 3, H, W)
        filepath: Path to save image
        nrow: Number of samples per row
        normalize: Whether to normalize images from [-1,1] to [0,1]
        title: Optional title for the image
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Create sample grid
    if normalize:
        samples = (samples + 1) / 2.0  # Convert from [-1,1] to [0,1]
    
    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    # Convert to numpy for matplotlib
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Plot and save
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    
    if title:
        plt.title(title, fontsize=16)
    
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Sample grid saved: {filepath}")

def interpolate_latent_codes(code1: torch.Tensor, code2: torch.Tensor, 
                           num_steps: int = 10) -> torch.Tensor:
    """
    Interpolate between two latent codes
    
    Args:
        code1: First latent code (1, latent_dim)
        code2: Second latent code (1, latent_dim)
        num_steps: Number of interpolation steps
        
    Returns:
        Interpolated codes (num_steps, latent_dim)
    """
    device = code1.device
    alphas = torch.linspace(0, 1, num_steps, device=device).view(-1, 1)
    
    interpolated = (1 - alphas) * code1 + alphas * code2
    return interpolated

def compute_fid_features(images: torch.Tensor, feature_extractor: nn.Module) -> torch.Tensor:
    """
    Compute FID features for a batch of images
    
    Args:
        images: Images tensor (batch_size, 3, H, W)
        feature_extractor: Pre-trained feature extractor (e.g., Inception-v3)
        
    Returns:
        Feature vectors (batch_size, feature_dim)
    """
    with torch.no_grad():
        # Resize images to 299x299 for Inception-v3
        if images.size(-1) != 299:
            images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear')
        
        # Extract features
        features = feature_extractor(images)
        
        # Flatten if necessary
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
    
    return features

def calculate_gradient_penalty(discriminator: nn.Module, real_images: torch.Tensor,
                             fake_images: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """
    Calculate gradient penalty for WGAN-GP
    
    Args:
        discriminator: Discriminator model
        real_images: Real images (batch_size, 3, H, W)
        fake_images: Fake images (batch_size, 3, H, W)
        device: Device to compute on
        
    Returns:
        Gradient penalty scalar
    """
    batch_size = real_images.size(0)
    
    # Random interpolation factor
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated images
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated images
    d_interpolated = discriminator.discriminate_only(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

def log_model_info(generator: nn.Module, discriminator: nn.Module):
    """
    Log information about the models
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
    """
    print("=== Model Information ===")
    
    # Generator info
    g_total, g_trainable = count_parameters(generator)
    print(f"Generator:")
    print(f"  Total parameters: {g_total:,}")
    print(f"  Trainable parameters: {g_trainable:,}")
    
    # Discriminator info
    d_total, d_trainable = count_parameters(discriminator)
    print(f"Discriminator:")
    print(f"  Total parameters: {d_total:,}")
    print(f"  Trainable parameters: {d_trainable:,}")
    
    # Total
    total_params = g_total + d_total
    print(f"Total GAN parameters: {total_params:,}")
    
    # Memory estimate (rough)
    memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Approximate memory usage: {memory_mb:.1f} MB")

def create_pokemon_type_samples(type_mappings: Dict[str, int], num_samples_per_type: int = 4,
                               device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Create sample conditions for all Pokemon types
    
    Args:
        type_mappings: Dictionary mapping type names to indices
        num_samples_per_type: Number of samples per type
        device: Device to create tensors on
        
    Returns:
        Tuple of (type_tensor, stats_tensor, type_names_list)
    """
    type_names = list(type_mappings.keys())
    total_samples = len(type_names) * num_samples_per_type
    
    # Create type conditions
    type_tensor = torch.zeros(total_samples, len(type_mappings), device=device)
    type_names_list = []
    
    for i, type_name in enumerate(type_names):
        start_idx = i * num_samples_per_type
        end_idx = start_idx + num_samples_per_type
        
        type_tensor[start_idx:end_idx, type_mappings[type_name]] = 1.0
        type_names_list.extend([type_name] * num_samples_per_type)
    
    # Create random stats
    stats_tensor = create_random_stats_tensor(total_samples, device=device)
    
    return type_tensor, stats_tensor, type_names_list

def test_model_utils():
    """Test model utilities"""
    print("Testing model utilities...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test type mappings
    type_mappings = {
        'Fire': 0, 'Water': 1, 'Grass': 2, 'Lightning': 3, 'Psychic': 4,
        'Fighting': 5, 'Darkness': 6, 'Metal': 7, 'Fairy': 8, 'Colorless': 9, 'Dragon': 10
    }
    
    # Test type condition creation
    type_names = ['Fire', 'Water', 'Grass']
    type_tensor = create_type_condition_tensor(type_names, type_mappings, device)
    print(f"✓ Type tensor shape: {type_tensor.shape}")
    
    # Test stats creation
    stats_tensor = create_random_stats_tensor(3, device=device)
    print(f"✓ Stats tensor shape: {stats_tensor.shape}")
    
    # Test interpolation
    code1 = torch.randn(1, 128, device=device)
    code2 = torch.randn(1, 128, device=device)
    interpolated = interpolate_latent_codes(code1, code2, 5)
    print(f"✓ Interpolated codes shape: {interpolated.shape}")
    
    # Test Pokemon type samples
    type_tensor, stats_tensor, type_names_list = create_pokemon_type_samples(
        type_mappings, num_samples_per_type=2, device=device
    )
    print(f"✓ All types tensor shape: {type_tensor.shape}")
    print(f"✓ Type names list length: {len(type_names_list)}")
    
    # Test sample grid saving (with dummy data)
    dummy_samples = torch.randn(8, 3, 64, 64)
    save_sample_grid(dummy_samples, "test_samples.png", nrow=4, title="Test Samples")
    
    print("Model utilities test completed!")

if __name__ == "__main__":
    test_model_utils()