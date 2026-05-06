import os
import random
from pathlib import Path

def load_dataset(file_path):
    """
    Load a dataset file and return list of (image_path, label) tuples.
    
    Args:
        file_path: Path to the train_list.txt file
        
    Returns:
        List of tuples: [(image_path, label), ...]
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    image_path, label = parts
                    data.append((image_path, label))
    return data

def save_dataset(data, output_path):
    """
    Save dataset to file in PaddleOCR format.
    
    Args:
        data: List of (image_path, label) tuples
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_path, label in data:
            f.write(f"{image_path}\t{label}\n")

def main():
    # Configuration
    train_ratio = 0.8  # 80% for training, 20% for testing
    random_seed = 42   # For reproducible splits
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Define dataset paths
    datasets = [
        {
            'name': 'long_text_config',
            'path': 'output/ppocr_format/long_text_config.txt'
        },
        {
            'name': 'short_text_config',
            'path': 'output/ppocr_format/short_text_config.txt'
        },
        {
            'name': 'number_text_config',
            'path': 'output/ppocr_format/number_text_config.txt'
        }
    ]
    
    # Load all datasets
    all_data = []
    dataset_stats = {}
    
    print("Loading datasets...")
    for dataset in datasets:
        if os.path.exists(dataset['path']):
            data = load_dataset(dataset['path'])
            all_data.extend(data)
            dataset_stats[dataset['name']] = len(data)
            print(f"  {dataset['name']}: {len(data)} samples")
        else:
            print(f"  Warning: {dataset['path']} not found, skipping...")
    
    print(f"\nTotal samples: {len(all_data)}")
    
    # Shuffle the combined dataset
    print("Shuffling combined dataset...")
    random.shuffle(all_data)
    
    # Split into train and test
    split_index = int(len(all_data) * train_ratio)
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Save train and test datasets
    output_dir = 'output/ppocr_format'
    train_path = os.path.join(output_dir, 'train_list.txt')
    test_path = os.path.join(output_dir, 'test_list.txt')
    
    print(f"\nSaving train dataset to: {train_path}")
    save_dataset(train_data, train_path)
    
    print(f"Saving test dataset to: {test_path}")
    save_dataset(test_data, test_path)
    
    # Show some sample entries
    print(f"\nSample train entries:")
    for i, (image_path, label) in enumerate(train_data[:5]):
        print(f"  {i+1}. {image_path}\t{label}")
    
    print(f"\nSample test entries:")
    for i, (image_path, label) in enumerate(test_data[:5]):
        print(f"  {i+1}. {image_path}\t{label}")
    
    print(f"\nSplit completed successfully!")
    print(f"Use {train_path} for training and {test_path} for evaluation.")

if __name__ == "__main__":
    main()
