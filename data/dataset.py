import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np

def pokemon_collate_fn(batch):
    """Custom collate function to handle variable-length metadata"""
    images = torch.stack([item['image'] for item in batch])
    types = torch.stack([item['types'] for item in batch])
    stats = torch.stack([item['stats'] for item in batch])
    
    metadata = {
        'id': [item['metadata']['id'] for item in batch],
        'name': [item['metadata']['name'] for item in batch], 
        'types_list': [item['metadata']['types_list'] for item in batch]
    }
    
    return {
        'image': images,
        'types': types,
        'stats': stats,
        'metadata': metadata
    }

class PokemonCardDataset(Dataset):
    def __init__(self, split='train', data_dir='processed_pokemon_data', transform=None):
        self.data_dir = data_dir
        self.split = split
        
        with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)
        
        with open(os.path.join(data_dir, 'metadata', f'{split}_split.json'), 'r') as f:
            self.data = json.load(f)
        
        with open(os.path.join(data_dir, 'metadata', 'type_mappings.json'), 'r') as f:
            mappings = json.load(f)
            self.type_to_idx = mappings['type_to_idx']
            self.num_types = mappings['num_types']
        
        with open(os.path.join(data_dir, 'metadata', 'stats_normalization.json'), 'r') as f:
            self.stats = json.load(f)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform
        
        print(f"Loaded {split} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        card = self.data[idx]
        
        artwork_path = card['artwork_path']
        try:
            image = Image.open(artwork_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {artwork_path}: {e}")
            image = torch.zeros(3, 256, 256)
        
        type_vector = torch.zeros(self.num_types)
        for ptype in card['types']:
            if ptype in self.type_to_idx:
                type_vector[self.type_to_idx[ptype]] = 1.0
        
        stats_vector = self._process_stats(card)
        
        return {
            'image': image,
            'types': type_vector,
            'stats': stats_vector,
            'metadata': {
                'id': card['id'],
                'name': card['name'],
                'types_list': card['types']
            }
        }
    
    def _process_stats(self, card):
        """Process card stats into normalized vector"""
        hp = card['hp'] if card['hp'] > 0 else self.stats['hp_mean']
        hp_norm = (hp - self.stats['hp_mean']) / self.stats['hp_std']
        
        num_attacks = len(card['attacks']) / max(self.stats['max_attacks'], 1)
        num_abilities = len(card['abilities']) / max(self.stats['max_abilities'], 1)
        
        rarity = card['rarity']
        if rarity in self.stats['rarity_order']:
            rarity_norm = self.stats['rarity_order'].index(rarity) / max(len(self.stats['rarity_order']) - 1, 1)
        else:
            rarity_norm = 0.5  
        
        stats = torch.tensor([hp_norm, num_attacks, num_abilities, rarity_norm], dtype=torch.float32)
        return stats

class PokemonDataModule:
    def __init__(self, data_dir='processed_pokemon_data', batch_size=32, num_workers=4): 
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)
    
    def train_dataloader(self):
        dataset = PokemonCardDataset('train', self.data_dir)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  
            collate_fn=pokemon_collate_fn 
        )
    
    def val_dataloader(self):
        dataset = PokemonCardDataset('val', self.data_dir)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pokemon_collate_fn 
        )
    
    def test_dataloader(self):
        dataset = PokemonCardDataset('test', self.data_dir)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pokemon_collate_fn 
        )
    
    def get_data_info(self):
        """Get information needed for model initialization"""
        return {
            'num_types': self.dataset_info['num_types'],
            'type_to_idx': self.dataset_info['type_to_idx'],
            'stats_dim': 4,
            'image_size': self.dataset_info['image_size']
        }

def test_dataset():
    """Test the dataset loading"""
    print("Testing Pokemon dataset...")
    
    data_module = PokemonDataModule(batch_size=8)
    
    train_loader = data_module.train_dataloader()
    print(f"Train batches: {len(train_loader)}")
    
    sample_batch = next(iter(train_loader))
    
    print(f"\nSample batch shapes:")
    print(f"  Images: {sample_batch['image'].shape}")
    print(f"  Types: {sample_batch['types'].shape}")
    print(f"  Stats: {sample_batch['stats'].shape}")
    
    print(f"\nSample data:")
    batch_size = sample_batch['image'].shape[0] 
    names = sample_batch['metadata']['name']
    types_lists = sample_batch['metadata']['types_list']
    
    for i in range(min(3, batch_size)):
        name = names[i]
        types = types_lists[i]
        print(f"  {i+1}. {name} - Types: {types}")
    
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    data_info = data_module.get_data_info()
    print(f"\nData info for model:")
    print(f"  Image size: {data_info['image_size']}")
    print(f"  Number of types: {data_info['num_types']}")
    print(f"  Stats dimensions: {data_info['stats_dim']}")
    
    print("\nDataset test completed successfully!")
    return data_module, data_info

if __name__ == "__main__":
    test_dataset()