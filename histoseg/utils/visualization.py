"""Visualization utilities for segmentation results."""

import colorsys
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def overlay_mask(image: Union[Image.Image, np.ndarray], 
                mask: Union[np.ndarray, torch.Tensor],
                colors: Optional[List[Tuple[int, int, int]]] = None,
                alpha: float = 0.5) -> Image.Image:
    """Overlay segmentation mask on image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Generate colors if not provided
    if colors is None:
        num_classes = int(mask.max()) + 1
        colors = generate_colors(num_classes)
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(1, len(colors)):  # Skip background (0)
        colored_mask[mask == class_id] = colors[class_id]
    
    colored_mask = Image.fromarray(colored_mask)
    
    # Blend with original image
    result = Image.blend(image, colored_mask, alpha)
    return result


def visualize_prediction(image: Union[Image.Image, np.ndarray],
                        prediction: Union[np.ndarray, torch.Tensor],
                        ground_truth: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        class_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """Visualize segmentation prediction with optional ground truth."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Setup figure
    num_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='tab20')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='tab20')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_legend(class_names: List[str], 
                 colors: List[Tuple[int, int, int]],
                 save_path: Optional[str] = None) -> Image.Image:
    """Create a legend for class colors."""
    # Calculate dimensions
    font_size = 20
    box_size = 30
    margin = 10
    
    width = max(len(name) for name in class_names) * font_size // 2 + box_size + margin * 3
    height = len(class_names) * (box_size + margin) + margin
    
    # Create image
    legend = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(legend)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw legend items
    for i, (name, color) in enumerate(zip(class_names, colors)):
        y = margin + i * (box_size + margin)
        
        # Draw color box
        draw.rectangle([margin, y, margin + box_size, y + box_size], fill=color)
        
        # Draw text
        draw.text((margin * 2 + box_size, y + box_size // 4), name, fill='black', font=font)
    
    if save_path:
        legend.save(save_path)
    
    return legend
