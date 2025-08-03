#!/usr/bin/env python3
"""
Quick inference script for testing trained models.

Usage:
    python scripts/inference.py --model_path checkpoints/model.ckpt \
                               --image_path test_image.jpg \
                               --output_path result.png
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from histoseg.training.mask2former_module import Mask2FormerModule
from histoseg.utils.visualization import overlay_mask, generate_colors


def load_model(checkpoint_path: str) -> Mask2FormerModule:
    """Load trained model from checkpoint."""
    model = Mask2FormerModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def preprocess_image(image_path: str, image_size: tuple = (512, 512)) -> torch.Tensor:
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply transforms
    transform = A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image


def inference(model: Mask2FormerModule, image_tensor: torch.Tensor) -> np.ndarray:
    """Run inference on image."""
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
            image_tensor = image_tensor.cuda()
        
        # Forward pass
        logits = model(image_tensor)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
    return predictions[0]  # Remove batch dimension


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_path", default="output.png", help="Path to save result")
    parser.add_argument("--image_size", nargs=2, type=int, default=[512, 512], 
                       help="Image size for inference (H W)")
    parser.add_argument("--overlay_alpha", type=float, default=0.5, 
                       help="Alpha for overlay visualization")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    print(f"Processing image {args.image_path}")
    image_tensor, original_image = preprocess_image(args.image_path, tuple(args.image_size))
    
    print("Running inference...")
    prediction = inference(model, image_tensor)
    
    # Generate visualization
    print("Creating visualization...")
    num_classes = model.num_classes
    colors = generate_colors(num_classes)
    
    # Resize prediction to original image size
    original_size = original_image.shape[:2]
    prediction_resized = np.array(Image.fromarray(prediction).resize(
        (original_size[1], original_size[0]), Image.NEAREST))
    
    # Create overlay
    original_pil = Image.fromarray(original_image)
    result = overlay_mask(original_pil, prediction_resized, colors, args.overlay_alpha)
    
    # Save result
    result.save(args.output_path)
    print(f"Result saved to {args.output_path}")
    
    # Print statistics
    unique_classes = np.unique(prediction)
    print(f"Detected classes: {unique_classes}")
    for class_id in unique_classes:
        pixel_count = np.sum(prediction == class_id)
        percentage = pixel_count / prediction.size * 100
        print(f"  Class {class_id}: {pixel_count} pixels ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
