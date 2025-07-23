# Pokemon Card GAN 🎮⚡

A Generative Adversarial Network that creates Pokemon trading card artwork with conditional generation based on Pokemon types and stats.

## 🚀 Quick Start

```bash
# 1. Collect Pokemon card data (optional - you have this)
python data/scraper.py

# 2. Preprocess data for training (optional - you have this)
python data/preprocessor.py

```


## 🎯 Features

- **Conditional Generation**: Generate Pokemon by type (Fire, Water, Grass, etc.)
- **Multi-Modal Conditioning**: Control HP, rarity, and abilities
- **High Resolution**: 256x256 Pokemon artwork generation
- **Progressive Training**: Resume from checkpoints
- **Real-time Monitoring**: Loss curves and sample generation

## 📊 Dataset

- **1040 Pokemon cards** from official TCG sets
- **11 Pokemon types** for conditional generation
- **256x256 extracted artwork** from full cards
- **Balanced train/val/test splits** (80/10/10%)

## 🏋️ Training

**Basic Training:**
```bash
python train_pokemon_gan.py
```

**Extended Training:**
```bash
python train_pokemon_gan.py --epochs 500 --batch_size 64
```

**Resume Training:**
```bash
python train_pokemon_gan.py --resume outputs/checkpoints/pokemon_gan_epoch_100.pth
```

**Training Options:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0002)
- `--patch_gan`: Use PatchGAN discriminator
- `--resume`: Resume from checkpoint

## 🎨 Model Architecture

**Generator (4.2M parameters):**
- Conditional batch normalization for type control
- Progressive upsampling: 4x4 → 256x256
- Type + stats conditioning

**Discriminator (45.9M parameters):**
- Auxiliary losses for type/stats prediction
- Spectral normalization for stability
- PatchGAN option for high-res generation
