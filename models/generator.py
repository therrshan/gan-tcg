import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization for type conditioning"""
    def __init__(self, num_features: int, num_conditions: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        # Learnable scale and bias conditioned on Pokemon type
        self.embed_scale = nn.Linear(num_conditions, num_features)
        self.embed_bias = nn.Linear(num_conditions, num_features)
        
        # Initialize to standard BN behavior
        nn.init.ones_(self.embed_scale.weight)
        nn.init.zeros_(self.embed_scale.bias)
        nn.init.zeros_(self.embed_bias.weight)
        nn.init.zeros_(self.embed_bias.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Standard batch norm (no affine transformation)
        out = self.bn(x)
        
        # Get conditional scale and bias
        scale = self.embed_scale(condition).view(-1, self.num_features, 1, 1)
        bias = self.embed_bias(condition).view(-1, self.num_features, 1, 1)
        
        # Apply conditional transformation
        return scale * out + bias

class GeneratorBlock(nn.Module):
    """Generator block with conditional batch norm and upsampling"""
    def __init__(self, in_channels: int, out_channels: int, num_conditions: int, 
                 upsample: bool = True):
        super().__init__()
        
        self.upsample = upsample
        
        # Main convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Conditional batch norm
        self.cbn = ConditionalBatchNorm2d(out_channels, num_conditions)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
        
        # Upsampling
        if upsample:
            self.upsample_layer = nn.ConvTranspose2d(
                out_channels, out_channels, 4, stride=2, padding=1
            )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Convolution
        x = self.conv(x)
        
        # Conditional batch normalization
        x = self.cbn(x, condition)
        
        # Activation
        x = self.activation(x)
        
        # Upsampling
        if self.upsample:
            x = self.upsample_layer(x)
        
        return x

class PokemonGenerator(nn.Module):
    """
    Conditional Generator for Pokemon card artwork
    
    Generates 256x256 Pokemon artwork conditioned on:
    - Pokemon type (Fire, Water, Grass, etc.)
    - Stats (HP, rarity, abilities)
    """
    
    def __init__(self, 
                 latent_dim: int = 128,
                 num_types: int = 11, 
                 stats_dim: int = 4,
                 img_channels: int = 3,
                 base_channels: int = 512):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_types = num_types
        self.stats_dim = stats_dim
        self.base_channels = base_channels
        
        # Total condition dimension (types + stats)
        self.condition_dim = num_types + stats_dim
        
        # Project noise and conditions to initial feature map
        # Start with 4x4 feature map
        self.initial_size = 4
        initial_features = base_channels * (self.initial_size ** 2)
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim + self.condition_dim, initial_features),
            nn.ReLU(inplace=True)
        )
        
        # Generator blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.blocks = nn.ModuleList([
            # 4x4 -> 8x8
            GeneratorBlock(base_channels, base_channels // 2, self.condition_dim),
            # 8x8 -> 16x16  
            GeneratorBlock(base_channels // 2, base_channels // 4, self.condition_dim),
            # 16x16 -> 32x32
            GeneratorBlock(base_channels // 4, base_channels // 8, self.condition_dim),
            # 32x32 -> 64x64
            GeneratorBlock(base_channels // 8, base_channels // 16, self.condition_dim),
            # 64x64 -> 128x128
            GeneratorBlock(base_channels // 16, base_channels // 32, self.condition_dim),
            # 128x128 -> 256x256
            GeneratorBlock(base_channels // 32, base_channels // 32, self.condition_dim),
        ])
        
        # Final layer to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_channels // 32, img_channels, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier normal"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, noise: torch.Tensor, types: torch.Tensor, 
                stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator
        
        Args:
            noise: Random noise vector (batch_size, latent_dim)
            types: One-hot type vector (batch_size, num_types) 
            stats: Normalized stats vector (batch_size, stats_dim)
            
        Returns:
            Generated Pokemon artwork (batch_size, 3, 256, 256)
        """
        batch_size = noise.size(0)
        
        # Combine all conditions
        conditions = torch.cat([types, stats], dim=1)  # (batch_size, condition_dim)
        
        # Combine noise and conditions
        input_vector = torch.cat([noise, conditions], dim=1)
        
        # Project to initial feature map
        x = self.projection(input_vector)
        x = x.view(batch_size, self.base_channels, self.initial_size, self.initial_size)
        
        # Pass through generator blocks
        for block in self.blocks:
            x = block(x, conditions)
        
        # Convert to RGB
        x = self.to_rgb(x)
        
        return x
    
    def generate_samples(self, num_samples: int, types: torch.Tensor, 
                        stats: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """
        Generate samples for inference
        
        Args:
            num_samples: Number of samples to generate
            types: Type conditions (num_samples, num_types)
            stats: Stats conditions (num_samples, stats_dim) 
            device: Device to generate on
            
        Returns:
            Generated samples (num_samples, 3, 256, 256)
        """
        self.eval()
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Move conditions to device
            types = types.to(device)
            stats = stats.to(device)
            
            # Generate samples
            samples = self.forward(noise, types, stats)
            
        return samples
    
    def interpolate_types(self, type1: torch.Tensor, type2: torch.Tensor,
                         stats: torch.Tensor, num_steps: int = 10,
                         device: str = 'cuda') -> torch.Tensor:
        """
        Interpolate between two Pokemon types
        
        Args:
            type1: First type vector (1, num_types)
            type2: Second type vector (1, num_types)  
            stats: Stats to use (1, stats_dim)
            num_steps: Number of interpolation steps
            device: Device to generate on
            
        Returns:
            Interpolated samples (num_steps, 3, 256, 256)
        """
        self.eval()
        with torch.no_grad():
            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=device).view(-1, 1)
            
            # Interpolate types (this creates a soft blend between types)
            interpolated_types = (1 - alphas) * type1 + alphas * type2
            
            # Repeat stats for all steps
            repeated_stats = stats.repeat(num_steps, 1)
            
            # Fixed noise for consistent comparison
            noise = torch.randn(1, self.latent_dim, device=device).repeat(num_steps, 1)
            
            # Generate interpolated samples
            samples = self.forward(noise, interpolated_types, repeated_stats)
            
        return samples

def test_generator():
    """Test the generator architecture"""
    print("Testing Pokemon Generator...")
    
    # Create generator
    generator = PokemonGenerator(
        latent_dim=128,
        num_types=11,
        stats_dim=4
    )
    
    # Test inputs
    batch_size = 8
    noise = torch.randn(batch_size, 128)
    types = torch.zeros(batch_size, 11)
    types[:, 0] = 1  # All Fire type for testing
    stats = torch.randn(batch_size, 4)
    
    # Forward pass
    with torch.no_grad():
        output = generator(noise, types, stats)
    
    print(f"✓ Generator output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test generation methods
    samples = generator.generate_samples(4, types[:4], stats[:4], device='cpu')
    print(f"✓ Sample generation shape: {samples.shape}")
    
    # Test interpolation
    type1 = torch.zeros(1, 11)
    type1[0, 0] = 1  # Fire
    type2 = torch.zeros(1, 11) 
    type2[0, 1] = 1  # Water
    
    interpolated = generator.interpolate_types(type1, type2, stats[:1], num_steps=5, device='cpu')
    print(f"✓ Type interpolation shape: {interpolated.shape}")
    
    print("Generator test completed successfully!")

if __name__ == "__main__":
    test_generator()