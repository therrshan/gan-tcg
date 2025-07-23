import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import random
from collections import Counter

class SimplePokemonDataPreprocessor:
    def __init__(self, 
                 json_path: str = "raw_pokemon_data/pokemon_data/pokemon_cards.json",
                 image_dir: str = "raw_pokemon_data/pokemon_image",  
                 output_dir: str = "processed_pokemon_data"):
        
        self.json_path = json_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/artwork", exist_ok=True)
        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        
        print(f"Processing Pokemon dataset...")
        print(f"Input: {json_path}")
        print(f"Output: {output_dir}")
    
    def load_and_clean_data(self):
        """Load JSON data - your data is already clean!"""
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
        
        print(f"Loaded {len(raw_data)} cards from JSON")
        
        cleaned_data = []
        for card in raw_data:
            image_path = card.get('image_path', '')
      
            possible_paths = [
                image_path,  
                image_path.replace('pokemon_image/', 'pokemon_images/'), 
                os.path.join('pokemon_images', os.path.basename(image_path)),  
                os.path.join('pokemon_image', os.path.basename(image_path))  
            ]
            
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            
            if not actual_path:
                print(f"⚠ Skipping {card.get('id', 'unknown')}: image not found")
                continue
            
            if not card.get('id') or not card.get('name'):
                continue
            
            hp = self._parse_hp(card.get('hp', '0'))
            
            types = card.get('types', ['Colorless'])
            
            cleaned_card = {
                'id': card['id'],
                'name': card['name'],
                'hp': hp,
                'types': types, 
                'rarity': card.get('rarity', 'Unknown'),
                'set': card.get('set', 'Unknown'),
                'image_path': actual_path,
                'attacks': card.get('attacks', []),
                'abilities': card.get('abilities', [])
            }
            
            cleaned_data.append(cleaned_card)
        
        print(f"Cleaned dataset: {len(cleaned_data)} valid cards")
        return cleaned_data
    
    def _parse_hp(self, hp_str):
        """Extract numeric HP from string"""
        if not hp_str or hp_str == 'N/A':
            return 0
        
        import re
        numbers = re.findall(r'\d+', str(hp_str))
        return int(numbers[0]) if numbers else 0
    
    def analyze_dataset(self, data):
        """Analyze dataset statistics"""
        print("\n=== Dataset Analysis ===")
        print(f"Total cards: {len(data)}")
        
        type_counts = Counter()
        for card in data:
            for ptype in card['types']:
                type_counts[ptype] += 1
        
        print(f"\nPokemon type distribution:")
        for ptype, count in type_counts.most_common():
            print(f"  {ptype}: {count}")
        
        print(f"\nSample cards:")
        for card in data[:5]:
            print(f"  {card['name']}: {card['types']} (HP: {card['hp']})")
        
        hp_values = [card['hp'] for card in data if card['hp'] > 0]
        if hp_values:
            print(f"\nHP statistics:")
            print(f"  Range: {min(hp_values)} - {max(hp_values)}")
            print(f"  Average: {np.mean(hp_values):.1f}")
        
        return type_counts
    
    def extract_artwork(self, data):
        """Extract Pokemon artwork from full card images"""
        print("\n=== Extracting Artwork ===")
        
        successful_extractions = 0
        
        for i, card in enumerate(data):
            try:
                img = Image.open(card['image_path']).convert('RGB')
                width, height = img.size
                
                left = int(width * 0.08)
                top = int(height * 0.12)
                right = int(width * 0.92)
                bottom = int(height * 0.55)
                
                artwork = img.crop((left, top, right, bottom))
                artwork_resized = artwork.resize((256, 256), Image.LANCZOS)
                
                artwork_path = f"{self.output_dir}/artwork/{card['id']}_artwork.jpg"
                artwork_resized.save(artwork_path, "JPEG", quality=95)
                
                full_card_resized = img.resize((224, 320), Image.LANCZOS)
                full_card_path = f"{self.output_dir}/images/{card['id']}.jpg"
                full_card_resized.save(full_card_path, "JPEG", quality=95)
                
                card['artwork_path'] = artwork_path
                card['processed_image_path'] = full_card_path
                
                successful_extractions += 1
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(data)} images...")
                    
            except Exception as e:
                print(f"Error processing {card['id']}: {e}")
                card['artwork_path'] = None
                card['processed_image_path'] = None
        
        print(f"Successfully extracted artwork from {successful_extractions}/{len(data)} cards")
        
        valid_data = [card for card in data if card.get('artwork_path')]
        print(f"Final dataset size: {len(valid_data)} cards with artwork")
        
        return valid_data
    
    def create_type_encodings(self, data):
        """Create type to index mappings"""
        all_types = set()
        for card in data:
            all_types.update(card['types'])
        
        sorted_types = sorted(all_types)
        type_to_idx = {ptype: idx for idx, ptype in enumerate(sorted_types)}
        idx_to_type = {idx: ptype for ptype, idx in type_to_idx.items()}
        
        mappings = {
            'type_to_idx': type_to_idx,
            'idx_to_type': idx_to_type,
            'num_types': len(sorted_types)
        }
        
        with open(f"{self.output_dir}/metadata/type_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"\nCreated type encodings for {len(sorted_types)} types:")
        print(f"Types: {sorted_types}")
        
        return mappings
    
    def normalize_stats(self, data):
        """Compute normalization parameters"""
        hp_values = [card['hp'] for card in data if card['hp'] > 0]
        hp_mean = np.mean(hp_values) if hp_values else 100
        hp_std = np.std(hp_values) if hp_values else 50
        
        attack_counts = [len(card['attacks']) for card in data]
        ability_counts = [len(card['abilities']) for card in data]
        
        max_attacks = max(attack_counts) if attack_counts else 2
        max_abilities = max(ability_counts) if ability_counts else 1
        
        rarity_order = ['Common', 'Uncommon', 'Rare', 'Rare Holo', 'Ultra Rare', 'Secret Rare', 'Promo']
        
        stats = {
            'hp_mean': float(hp_mean),
            'hp_std': float(hp_std),
            'max_attacks': max_attacks,
            'max_abilities': max_abilities,
            'rarity_order': rarity_order
        }
        
        with open(f"{self.output_dir}/metadata/stats_normalization.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nStats normalization:")
        print(f"  HP: mean={hp_mean:.1f}, std={hp_std:.1f}")
        print(f"  Max attacks: {max_attacks}, Max abilities: {max_abilities}")
        
        return stats
    
    def create_train_val_test_splits(self, data, train_ratio=0.8, val_ratio=0.1):
        """Create balanced dataset splits"""
        print(f"\n=== Creating Dataset Splits ===")
        
        random.seed(42)
        shuffled_data = random.sample(data, len(data))
        
        n_total = len(shuffled_data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        train_data = shuffled_data[:n_train]
        val_data = shuffled_data[n_train:n_train + n_val]
        test_data = shuffled_data[n_train + n_val:]
        
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        
        for split_name, split_data in splits.items():
            with open(f"{self.output_dir}/metadata/{split_name}_split.json", 'w') as f:
                json.dump(split_data, f, indent=2)
            
            df = pd.DataFrame(split_data)
            df.to_csv(f"{self.output_dir}/metadata/{split_name}_split.csv", index=False)
        
        print(f"Dataset splits created:")
        print(f"  Train: {len(train_data)} cards ({train_ratio*100:.0f}%)")
        print(f"  Val: {len(val_data)} cards ({val_ratio*100:.0f}%)")
        print(f"  Test: {len(test_data)} cards ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return splits
    
    def create_pytorch_dataset_info(self, type_mappings, stats):
        """Create info file for PyTorch dataset"""
        dataset_info = {
            'image_size': 256,
            'num_types': type_mappings['num_types'],
            'type_to_idx': type_mappings['type_to_idx'],
            'stats_normalization': stats,
            'data_dir': self.output_dir
        }
        
        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset info saved for PyTorch training")
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...\n")
        
        data = self.load_and_clean_data()
        
        type_counts = self.analyze_dataset(data)
        
        processed_data = self.extract_artwork(data)
        
        type_mappings = self.create_type_encodings(processed_data)
        
        stats = self.normalize_stats(processed_data)
        
        splits = self.create_train_val_test_splits(processed_data)
        
        self.create_pytorch_dataset_info(type_mappings, stats)
        
        print(f"\n=== Preprocessing Complete ===")
        print(f"Final dataset: {len(processed_data)} cards")
        print(f"Pokemon types: {list(type_mappings['type_to_idx'].keys())}")
        print(f"Ready for GAN training!")
        
        return processed_data, splits, type_mappings, stats

def verify_preprocessing(output_dir):
    """Verify preprocessing results"""
    print(f"\n=== Verification ===")
    
    required_files = [
        "dataset_info.json",
        "metadata/type_mappings.json", 
        "metadata/stats_normalization.json",
        "metadata/train_split.json",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(output_dir, file_path)
        status = "✓" if os.path.exists(full_path) else "✗"
        print(f"{status} {file_path}")
    
    if os.path.exists(f"{output_dir}/artwork"):
        artwork_count = len([f for f in os.listdir(f"{output_dir}/artwork") if f.endswith('.jpg')])
        print(f"Artwork images: {artwork_count}")
    
    try:
        with open(f"{output_dir}/metadata/train_split.json", 'r') as f:
            train_data = json.load(f)
        
        print(f"Train dataset: {len(train_data)} cards")
        
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"Sample card: {sample['name']} - Types: {sample['types']}")
    except:
        print("Could not verify dataset contents")
    
    print("\nDataset ready for training!")

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = SimplePokemonDataPreprocessor()
    data, splits, mappings, stats = preprocessor.run_full_preprocessing()
    
    # Verify results
    verify_preprocessing("processed_pokemon_data")