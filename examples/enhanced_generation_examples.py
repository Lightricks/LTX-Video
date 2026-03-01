"""
Examples demonstrating the enhanced LTX-Video features
"""

import torch
from PIL import Image
import numpy as np

from ltx_video.pipelines.pipeline_ltx_video_enhanced import LTXVideoEnhancedPipeline
from ltx_video.utils.motion_presets import MotionPresets, StylePresets, create_cinematic_shot, create_artistic_video


def example_motion_controlled_generation():
    """Example: Generate video with precise camera movements"""
    
    # Initialize enhanced pipeline
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    # Example 1: Camera pan with custom trajectory
    result = pipeline.generate_with_motion_control(
        prompt="A serene mountain landscape with a crystal clear lake reflecting the snow-capped peaks",
        motion_type="camera_pan",
        camera_trajectory={
            "pan": [-30, -15, 0, 15, 30],  # Pan from left to right
            "tilt": [0, 0, 0, 0, 0],       # Keep level
            "zoom": [1.0, 1.0, 1.2, 1.2, 1.0]  # Slight zoom in middle
        },
        height=704,
        width=1216,
        num_frames=121,
        num_inference_steps=30
    )
    
    # Example 2: Object trajectory
    result = pipeline.generate_with_motion_control(
        prompt="A red sports car driving through a winding mountain road",
        motion_type="object_trajectory",
        control_points=[
            (0.1, 0.7, 0.0),   # Start bottom left
            (0.3, 0.5, 0.25),  # Move up and right
            (0.7, 0.3, 0.75),  # Continue trajectory
            (0.9, 0.2, 1.0)    # End top right
        ]
    )
    
    # Example 3: Cinematic dolly zoom
    dolly_params = create_cinematic_shot("dolly_zoom", duration_frames=121)
    result = pipeline(
        prompt="A person standing in a crowded street, dramatic moment",
        **dolly_params
    )
    
    return result


def example_style_transfer_generation():
    """Example: Generate videos with different artistic styles"""
    
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    base_prompt = "A woman walking through a garden filled with blooming flowers"
    
    # Generate in different styles
    styles_to_try = ["anime", "oil_painting", "cyberpunk", "watercolor"]
    
    results = {}
    for style in styles_to_try:
        result = pipeline.generate_with_style(
            prompt=base_prompt,
            style_name=style,
            style_strength=0.8,
            height=704,
            width=1216,
            num_frames=121
        )
        results[style] = result
    
    # Create artistic video with motion
    artistic_params = create_artistic_video(
        style="impressionist",
        motion="smooth",
        duration_frames=121
    )
    
    artistic_result = pipeline(
        prompt="A sunset over a field of sunflowers, golden hour lighting",
        **artistic_params
    )
    
    return results, artistic_result


def example_multimodal_conditioning():
    """Example: Generate videos with multi-modal inputs"""
    
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    # Example 1: Depth-conditioned generation
    # Create a simple depth map (closer objects are brighter)
    depth_map = np.zeros((704, 1216))
    # Add a person silhouette in the center (closer)
    depth_map[200:500, 400:800] = 0.8
    # Add background (farther)
    depth_map[depth_map == 0] = 0.3
    
    depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)
    
    result = pipeline.depth_conditioned_generation(
        prompt="A person standing in a beautiful garden with depth and dimension",
        depth_image=depth_tensor,
        height=704,
        width=1216,
        num_frames=121
    )
    
    # Example 2: Pose-conditioned generation
    # Define human pose keypoints (17 keypoints in COCO format)
    pose_keypoints = [
        [0.5, 0.2],   # nose
        [0.48, 0.18], # left_eye
        [0.52, 0.18], # right_eye
        [0.46, 0.2],  # left_ear
        [0.54, 0.2],  # right_ear
        [0.4, 0.35],  # left_shoulder
        [0.6, 0.35],  # right_shoulder
        [0.35, 0.5],  # left_elbow
        [0.65, 0.5],  # right_elbow
        [0.3, 0.65],  # left_wrist
        [0.7, 0.65],  # right_wrist
        [0.45, 0.6],  # left_hip
        [0.55, 0.6],  # right_hip
        [0.43, 0.8],  # left_knee
        [0.57, 0.8],  # right_knee
        [0.41, 0.95], # left_ankle
        [0.59, 0.95]  # right_ankle
    ]
    
    result = pipeline.pose_conditioned_generation(
        prompt="A dancer performing graceful movements in a studio",
        pose_keypoints=pose_keypoints,
        height=704,
        width=1216,
        num_frames=121
    )
    
    # Example 3: Audio-to-video (placeholder)
    result = pipeline.audio_to_video(
        audio_path="path/to/music.wav",
        prompt="A music video with dynamic visuals synchronized to the beat",
        style_name="cyberpunk",
        motion_type="camera_pan"
    )
    
    return result


def example_temporal_consistency():
    """Example: Enhanced temporal consistency and frame interpolation"""
    
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    # Generate with enhanced temporal consistency
    result = pipeline(
        prompt="A cat gracefully jumping from one rooftop to another in slow motion",
        enable_temporal_consistency=True,
        interpolation_frames=[30, 60, 90],  # Add extra smoothness at these frames
        height=704,
        width=1216,
        num_frames=121,
        num_inference_steps=40
    )
    
    return result


def example_combined_features():
    """Example: Combining multiple enhancement features"""
    
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    # Create a complex scene with multiple enhancements
    motion_params = MotionPresets.camera_orbit(duration_frames=121, radius=3.0)
    
    result = pipeline(
        prompt="A futuristic cityscape at night with neon lights and flying cars",
        
        # Motion control
        **motion_params,
        
        # Style transfer
        style_name="cyberpunk",
        style_strength=0.9,
        
        # Temporal consistency
        enable_temporal_consistency=True,
        
        # Generation parameters
        height=704,
        width=1216,
        num_frames=121,
        num_inference_steps=40,
        guidance_scale=3.5
    )
    
    return result


def example_real_time_generation():
    """Example: Real-time generation with optimizations"""
    
    pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")
    
    # Optimize for speed
    pipeline.enable_model_cpu_offload()  # If available
    
    # Generate shorter clips for real-time use
    result = pipeline(
        prompt="A quick camera movement revealing a beautiful landscape",
        motion_type="camera_pan",
        style_name="photorealistic",
        height=512,  # Lower resolution for speed
        width=768,
        num_frames=65,  # Shorter duration
        num_inference_steps=20,  # Fewer steps for speed
        enable_temporal_consistency=False  # Disable for speed
    )
    
    return result


if __name__ == "__main__":
    # Run examples
    print("Running enhanced LTX-Video examples...")
    
    # Uncomment to run specific examples
    # motion_result = example_motion_controlled_generation()
    # style_results = example_style_transfer_generation()
    # multimodal_result = example_multimodal_conditioning()
    # temporal_result = example_temporal_consistency()
    # combined_result = example_combined_features()
    # realtime_result = example_real_time_generation()
    
    print("Examples completed!")