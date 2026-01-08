#!/usr/bin/env python3
"""
Download pre-trained models from HuggingFace Hub
Downloads BERT and DeiT models used in FL-EE experiments
"""

import os
import argparse
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification
)

# Define models to download
# Format: "display_name": "huggingface_model_id"
MODELS = {
    "BERT Models": {
        "bert-12-128": "google/bert_uncased_L-12_H-128_A-2",
        "bert-12-256": "google/bert_uncased_L-12_H-256_A-4",
    },
    "Vision Models": {
        "deit-tiny": "facebook/deit-tiny-patch16-224",
        "deit-small": "facebook/deit-small-patch16-224",
    }
}

# Default save directory
SAVE_DIR = "./models"


def download_bert_model(model_name, save_dir):
    """Download BERT model and tokenizer"""
    print(f"\n{'='*60}")
    print(f"Downloading BERT model: {model_name}")
    print(f"{'='*60}")
    
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print(f"  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print(f"  ✓ Tokenizer saved to {save_path}")
        
        # Download model
        print(f"  → Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Default for binary classification
            ignore_mismatched_sizes=True
        )
        model.save_pretrained(save_path)
        print(f"  ✓ Model saved to {save_path}")
        
        print(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False


def download_vision_model(model_name, save_dir):
    """Download Vision Transformer model and image processor"""
    print(f"\n{'='*60}")
    print(f"Downloading Vision model: {model_name}")
    print(f"{'='*60}")
    
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Download image processor
        print(f"  → Downloading image processor...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        processor.save_pretrained(save_path)
        print(f"  ✓ Image processor saved to {save_path}")
        
        # Download model
        print(f"  → Downloading model...")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )
        model.save_pretrained(save_path)
        print(f"  ✓ Model saved to {save_path}")
        
        print(f"✅ Successfully downloaded: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models from HuggingFace Hub")
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['bert-12-128', 'bert-12-256', 'deit-tiny', 'deit-small', 'all'],
        default=['all'],
        help='Models to download (default: all)'
    )
    parser.add_argument(
        '--save-dir',
        default='./models',
        help='Directory to save models (default: ./models)'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("FL-EE Model Downloader")
    print("="*60)
    
    save_dir = args.save_dir
    print(f"Models will be saved to: {os.path.abspath(save_dir)}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine which models to download
    selected_models = args.models
    if 'all' in selected_models:
        selected_models = list(MODELS["BERT Models"].keys()) + list(MODELS["Vision Models"].keys())
    
    print(f"Selected models: {', '.join(selected_models)}")
    
    success_count = 0
    total_count = 0
    
    # Download BERT models
    bert_models = [m for m in selected_models if m in MODELS["BERT Models"]]
    if bert_models:
        print("\n" + "="*60)
        print("DOWNLOADING BERT MODELS")
        print("="*60)
        for model_key in bert_models:
            model_name = MODELS["BERT Models"][model_key]
            total_count += 1
            if download_bert_model(model_name, save_dir):
                success_count += 1
    
    # Download Vision models
    vision_models = [m for m in selected_models if m in MODELS["Vision Models"]]
    if vision_models:
        print("\n" + "="*60)
        print("DOWNLOADING VISION MODELS")
        print("="*60)
        for model_key in vision_models:
            model_name = MODELS["Vision Models"][model_key]
            total_count += 1
            if download_vision_model(model_name, save_dir):
                success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successfully downloaded: {success_count}/{total_count} models")
    print(f"📁 Models saved to: {os.path.abspath(save_dir)}")
    
    if success_count < total_count:
        print(f"⚠️  {total_count - success_count} model(s) failed to download")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
