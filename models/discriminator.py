import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DiscriminatorBlock(nn.Module):
    """Discriminator block with downsampling and spectral normalization"""
    def __init__(self, in_channels: int, out_channels: int, 
                 downsample: bool = True, use_spectral_norm: bool = True):
        super().__init__()
        
        self.downsample = downsample
        
        # Main convolution
        conv = nn.Conv2d(in_channels, out_channels, 4, stride=2 if downsample else 1, padding=1)
        
        # Apply spectral normalization for training stability
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        
        self.conv = conv
        
        # Batch normalization (not for first layer)
        self.use_bn = in_channels != 3  # Don't use BN for RGB input
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        # Leaky ReLU activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.use_bn:
            x = self.bn(x)
        
        x = self.activation(x)
        return x

class PokemonDiscriminator(nn.Module):
    """
    Conditional Discriminator for Pokemon card artwork
    
    Discriminates between real and fake Pokemon artwork while also
    predicting Pokemon type and stats for auxiliary losses.
    """
    
    def __init__(self, 
                 img_channels: int = 3,
                 num_types: int = 11,
                 stats_dim: int = 4, 
                 base_channels: int = 64):
        super().__init__()
        
        self.num_types = num_types
        self.stats_dim = stats_dim
        self.base_channels = base_channels
        
        # Image encoder: 256x256 -> 4x4
        self.image_encoder = nn.Sequential(
            # 256x256 -> 128x128
            DiscriminatorBlock(img_channels, base_channels, downsample=True, use_spectral_norm=True),
            # 128x128 -> 64x64
            DiscriminatorBlock(base_channels, base_channels * 2, downsample=True),
            # 64x64 -> 32x32
            DiscriminatorBlock(base_channels * 2, base_channels * 4, downsample=True),
            # 32x32 -> 16x16
            DiscriminatorBlock(base_channels * 4, base_channels * 8, downsample=True),
            # 16x16 -> 8x8
            DiscriminatorBlock(base_channels * 8, base_channels * 16, downsample=True),
            # 8x8 -> 4x4
            DiscriminatorBlock(base_channels * 16, base_channels * 16, downsample=True),
        )
        
        # Feature dimension after image encoding
        self.feature_dim = base_channels * 16 * 4 * 4
        
        # Condition encoder (for real images)
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_types + stats_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Combined feature processing
        self.combined_features = nn.Sequential(
            nn.Linear(self.feature_dim + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output heads
        self.real_fake_head = nn.Linear(512, 1)  # Real/fake classification
        self.type_head = nn.Linear(512, num_types)  # Type prediction (auxiliary task)
        self.stats_head = nn.Linear(512, stats_dim)  # Stats prediction (auxiliary task)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier normal"""
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor, types: torch.Tensor = None, 
                stats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the discriminator
        
        Args:
            images: Input images (batch_size, 3, 256, 256)
            types: Type conditions (batch_size, num_types) - for real images
            stats: Stats conditions (batch_size, stats_dim) - for real images
            
        Returns:
            Tuple of (real_fake_logits, predicted_types, predicted_stats)
        """
        batch_size = images.size(0)
        
        # Encode images
        img_features = self.image_encoder(images)
        img_features = img_features.view(batch_size, -1)
        
        # Process conditions if provided (for real images)
        if types is not None and stats is not None:
            conditions = torch.cat([types, stats], dim=1)
            condition_features = self.condition_encoder(conditions)
        else:
            # For fake images, use zero conditions
            condition_features = torch.zeros(batch_size, 512, device=images.device)
        
        # Combine image and condition features
        combined = torch.cat([img_features, condition_features], dim=1)
        features = self.combined_features(combined)
        
        # Output predictions
        real_fake_logits = self.real_fake_head(features)
        predicted_types = self.type_head(features)
        predicted_stats = self.stats_head(features)
        
        return real_fake_logits, predicted_types, predicted_stats
    
    def discriminate_only(self, images: torch.Tensor) -> torch.Tensor:
        """
        Only perform real/fake discrimination (for generated images)
        
        Args:
            images: Input images (batch_size, 3, 256, 256)
            
        Returns:
            Real/fake logits (batch_size, 1)
        """
        real_fake_logits, _, _ = self.forward(images)
        return real_fake_logits

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator - classifies patches as real/fake
    Good for high-resolution image generation
    """
    
    def __init__(self, 
                 img_channels: int = 3,
                 num_types: int = 11,
                 stats_dim: int = 4,
                 base_channels: int = 64):
        super().__init__()
        
        self.num_types = num_types
        self.stats_dim = stats_dim
        
        # PatchGAN architecture - outputs 16x16 patch classifications
        self.discriminator = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(img_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 16x16 (final patch classification)
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1)
        )
        
        # Auxiliary classifiers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.type_classifier = nn.Sequential(
            nn.Linear(base_channels * 8, num_types)
        )
        self.stats_regressor = nn.Sequential(
            nn.Linear(base_channels * 8, stats_dim)
        )
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of PatchGAN discriminator
        
        Args:
            images: Input images (batch_size, 3, 256, 256)
            
        Returns:
            Tuple of (patch_logits, predicted_types, predicted_stats)
        """
        # Get intermediate features before final layer
        x = images
        for layer in self.discriminator[:-1]:
            x = layer(x)
        
        # Global features for auxiliary tasks
        global_features = self.global_pool(x).squeeze(-1).squeeze(-1)
        predicted_types = self.type_classifier(global_features)
        predicted_stats = self.stats_regressor(global_features)
        
        # Patch-wise real/fake classification
        patch_logits = self.discriminator[-1](x)
        
        return patch_logits, predicted_types, predicted_stats

def test_discriminators():
    """Test both discriminator architectures"""
    print("Testing Pokemon Discriminators...")
    
    # Test regular discriminator
    print("\n--- Testing Regular Discriminator ---")
    discriminator = PokemonDiscriminator(
        num_types=11,
        stats_dim=4
    )
    
    # Test inputs
    batch_size = 8
    images = torch.randn(batch_size, 3, 256, 256)
    types = torch.zeros(batch_size, 11)
    types[:, 0] = 1  # Fire type
    stats = torch.randn(batch_size, 4)
    
    # Forward pass with conditions (real images)
    real_fake_logits, pred_types, pred_stats = discriminator(images, types, stats)
    
    print(f"✓ Real/fake logits shape: {real_fake_logits.shape}")
    print(f"✓ Predicted types shape: {pred_types.shape}")
    print(f"✓ Predicted stats shape: {pred_stats.shape}")
    
    # Forward pass without conditions (fake images)
    fake_logits = discriminator.discriminate_only(images)
    print(f"✓ Fake-only logits shape: {fake_logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"✓ Regular D parameters: {total_params:,}")
    
    # Test PatchGAN discriminator
    print("\n--- Testing PatchGAN Discriminator ---")
    patch_discriminator = PatchGANDiscriminator(
        num_types=11,
        stats_dim=4
    )
    
    patch_logits, patch_types, patch_stats = patch_discriminator(images)
    
    print(f"✓ Patch logits shape: {patch_logits.shape}")
    print(f"✓ Patch types shape: {patch_types.shape}")
    print(f"✓ Patch stats shape: {patch_stats.shape}")
    
    # Count parameters
    patch_params = sum(p.numel() for p in patch_discriminator.parameters())
    print(f"✓ PatchGAN D parameters: {patch_params:,}")
    
    print("\nDiscriminator tests completed successfully!")

if __name__ == "__main__":
    test_discriminators()