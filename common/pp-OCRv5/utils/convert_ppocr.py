#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert text_renderer generated data to PaddleOCR training format.

This script converts the output from text_renderer to the format that PaddleOCR expects:
image_path\ttext_content

Usage:
    python convert_ppocr.py
"""

import os
import json
import argparse
from pathlib import Path


def convert_config_to_ppocr(config_dir, output_file):
    """
    Convert a single config directory to PaddleOCR format.
    
    Args:
        config_dir (str): Path to the config directory (e.g., 'short_text_config')
        output_file (str): Output file path for the converted data
    """
    config_path = Path(config_dir)
    labels_file = config_path / "labels.json"
    images_dir = config_path / "images"
    
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found, skipping {config_dir}")
        return
    
    if not images_dir.exists():
        print(f"Warning: {images_dir} not found, skipping {config_dir}")
        return
    
    print(f"Processing {config_dir}...")
    
    # Read labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data.get('labels', {})
    
    # Convert to PaddleOCR format
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_id, text in labels.items():
            # Construct image path with config folder name to avoid confusion
            config_name = config_path.name
            image_path = f"{config_name}/images/{image_id}.jpg"
            
            # Write in PaddleOCR format: image_path\ttext
            f.write(f"{image_path}\t{text}\n")
    
    print(f"  Converted {len(labels)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert text_renderer data to PaddleOCR format')
    parser.add_argument('--input_dir', default='output/render', 
                       help='Input directory containing the config folders (default: output/render)')
    parser.add_argument('--output_dir', default='output/ppocr_format',
                       help='Output directory for converted files (default: output/ppocr_format)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all config directories
    config_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.endswith('_config'):
            config_dirs.append(item.name)
    
    if not config_dirs:
        print(f"No config directories found in {input_dir}")
        return
    
    print(f"Found config directories: {config_dirs}")
    
    # Convert each config
    for config_dir in config_dirs:
        config_path = input_dir / config_dir
        output_file = output_dir / f"{config_dir}.txt"
        
        convert_config_to_ppocr(config_path, output_file)
    
    print(f"Conversion completed! Output files saved to: {output_dir}")
    print(f"Individual files:")
    for config_dir in config_dirs:
        print(f"  - {config_dir}.txt")


if __name__ == "__main__":
    main()
