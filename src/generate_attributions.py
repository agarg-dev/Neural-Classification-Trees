#!/usr/bin/env python3
"""
Standalone script to regenerate attribution visualizations.
ALWAYS saves to a unique 'run_id' directory (Job_ID + Timestamp).
"""

import argparse
import os
import sys
import pandas as pd
import torch
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Add parent directory to path to import 'nct' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nct import NCTClassifier

# Updated list to match your new class capabilities
SUPPORTED_METHODS = [
    "layer_gradcam", "guided_gradcam", "layer_conductance",
    "layer_integrated_gradients", "layer_gradient_x_activation",
    "saliency", "input_x_gradient", "deeplift", 
    "guided_backprop", "occlusion"
]

class WaterbirdDataset(Dataset):
    """PyTorch Dataset for loading Waterbird images."""
    def __init__(self, metadata_df: pd.DataFrame, root_dir: str, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.metadata.loc[idx, 'img_filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.metadata.loc[idx, 'y'])
        if self.transform:
            image = self.transform(image)
        return image, label

def load_waterbird_data(data_dir: str):
    dataset_path = os.path.join(data_dir, "waterbird")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    metadata_path = os.path.join(dataset_path, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    # Use split 2 (Test Set)
    test_metadata = metadata[metadata['split'] == 2]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = WaterbirdDataset(test_metadata, dataset_path, transform=transform)
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    return test_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--data_dir', type=str, default='./data', help="Root data directory")
    parser.add_argument('--attribution_dir', type=str, default='./attributions', help="Root output directory")
    parser.add_argument('--num_per_node', type=int, default=5, help="Samples per node")
    parser.add_argument('--num_transitions', type=int, default=5, help="Transition samples")
    
    # Updated help text to reflect new keywords
    parser.add_argument('--methods', type=str, default='all_layer',
                       help='Comma-separated list or "all", "all_layer", "all_pixel".')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. ALWAYS create Unique Run Directory
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = os.path.join(args.attribution_dir, f"{job_id}_{timestamp}")

    print("=" * 70)
    print(f"Global Output Directory: {base_run_dir}")
    print("=" * 70)

    # 2. Determine Methods (Parse argument for the class)
    raw_input = args.methods.lower().strip()
    
    # Check for special keywords
    if raw_input in ['all', 'all_layer', 'all_pixel']:
        methods_arg = raw_input
    else:
        # Create a list of strings
        methods_arg = [m.strip() for m in raw_input.split(',')]

    # 3. Load Data and Model
    try:
        test_dataset = load_waterbird_data(args.data_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {args.checkpoint}")
        classifier = NCTClassifier.from_checkpoint(args.checkpoint, device=device)
        print(f"Model loaded on {device}.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model or data: {e}")
        return

    # 4. Run Visualization (Delegate to Class)
    # Iteration results are loaded from checkpoint and stored in classifier._iteration_results
    try:
        classifier.generate_attribution_visualizations(
            X_test=test_dataset,
            save_dir=base_run_dir,
            num_per_node=args.num_per_node,
            num_transitions=args.num_transitions,
            methods=methods_arg  # Pass the list or keyword here
        )
    except Exception as e:
        print(f">>> ERROR running visualizations: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Script Completed. Output: {base_run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()