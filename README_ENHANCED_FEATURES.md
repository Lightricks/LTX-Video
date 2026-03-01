# LTX-Video Enhanced Features

This document describes the new advanced features added to the LTX-Video model for enhanced video generation capabilities.

## 🎬 New Features Overview

### 1. Motion Control System
- **Precise Camera Movements**: Pan, tilt, zoom, orbit, and dolly shots
- **Object Trajectory Control**: Define custom paths for objects in the scene
- **Cinematic Presets**: Pre-built camera movements for professional shots
- **Real-time Motion Adjustment**: Dynamic motion parameter changes during generation

### 2. Style Transfer & Artistic Effects
- **Multiple Art Styles**: Anime, oil painting, watercolor, cyberpunk, vintage film, noir
- **Adjustable Style Strength**: Fine-tune the intensity of artistic effects
- **Real-time Style Switching**: Change styles during video generation
- **Custom Style Training**: Support for training custom artistic styles

### 3. Multi-Modal Conditioning
- **Audio-to-Video**: Generate videos synchronized with audio input
- **Depth-Conditioned Generation**: Use depth maps to control scene geometry
- **Pose-Conditioned Generation**: Generate videos following human pose sequences
- **Semantic Segmentation**: Control generation with semantic layout maps

### 4. Enhanced Temporal Consistency
- **Advanced Frame Interpolation**: Smoother motion between keyframes
- **Temporal Attention**: Better consistency across video sequences
- **Optical Flow Integration**: Physics-aware motion generation
- **Adaptive Consistency**: Adjust consistency based on scene complexity

### 5. Real-Time Optimizations
- **Streaming Generation**: Generate video chunks for real-time playback
- **Adaptive Quality Scaling**: Automatically adjust quality based on performance
- **Memory Optimizations**: Efficient memory usage for longer videos
- **Quality Presets**: Pre-configured settings for different use cases

## 🚀 Quick Start Examples

### Basic Motion Control
```python
from ltx_video.pipelines.pipeline_ltx_video_enhanced import LTXVideoEnhancedPipeline
from ltx_video.utils.motion_presets import create_cinematic_shot

# Load enhanced pipeline
pipeline = LTXVideoEnhancedPipeline.from_pretrained("path/to/ltx-video-model")

# Generate with camera pan
result = pipeline.generate_with_motion_control(
    prompt="A serene mountain landscape with a crystal clear lake",
    motion_type="camera_pan",
    camera_trajectory={
        "pan": [-30, 0, 30],  # Pan from left to right
        "zoom": [1.0, 1.2, 1.0]  # Slight zoom in middle
    }
)

# Use cinematic preset
dolly_params = create_cinematic_shot("dolly_zoom")
result = pipeline(
    prompt="A dramatic moment in a crowded street",
    **dolly_params
)
```

### Style Transfer
```python
# Generate in anime style
result = pipeline.generate_with_style(
    prompt="A woman walking through a garden",
    style_name="anime",
    style_strength=0.8
)

# Available styles: photorealistic, anime, oil_painting, watercolor, 
# sketch, cyberpunk, vintage_film, noir
```

### Multi-Modal Generation
```python
# Audio-to-video generation
result = pipeline.audio_to_video(
    audio_path="music.wav",
    prompt="A music video with dynamic visuals",
    style_name="cyberpunk"
)

# Depth-conditioned generation
result = pipeline.depth_conditioned_generation(
    prompt="A person in a detailed environment",
    depth_image="depth_map.png"
)

# Pose-conditioned generation
pose_keypoints = [[0.5, 0.2], [0.48, 0.18], ...]  # 17 keypoints
result = pipeline.pose_conditioned_generation(
    prompt="A dancer performing graceful movements",
    pose_keypoints=pose_keypoints
)
```

### Real-Time Generation
```python
from ltx_video.utils.real_time_optimizations import setup_real_time_pipeline, QualityPresets

# Set up optimized pipeline
pipeline, optimizer = setup_real_time_pipeline("path/to/model", "speed")

# Real-time preview generation
preview_settings = QualityPresets.real_time_preview()
result = pipeline(
    prompt="Quick preview of a sunset scene",
    **preview_settings
)

# Streaming generation for real-time playback
for chunk in optimizer.generate_streaming(
    prompt="A long scenic drive through mountains",
    chunk_frames=32,
    num_frames=257
):
    # Process chunk for real-time display
    display_video_chunk(chunk)
```

## 🎯 Advanced Usage

### Combining Multiple Features
```python
# Complex scene with multiple enhancements
from ltx_video.utils.motion_presets import MotionPresets

motion_params = MotionPresets.camera_orbit(radius=3.0)

result = pipeline(
    prompt="A futuristic cityscape at night",
    
    # Motion control
    **motion_params,
    
    # Style transfer
    style_name="cyberpunk",
    style_strength=0.9,
    
    # Enhanced temporal consistency
    enable_temporal_consistency=True,
    interpolation_frames=[30, 60, 90],
    
    # Quality settings
    height=704,
    width=1216,
    num_frames=121,
    num_inference_steps=40
)
```

### Custom Motion Trajectories
```python
# Define custom object trajectory
control_points = [
    (0.1, 0.7, 0.0),   # Start position (x, y, time)
    (0.3, 0.5, 0.25),  # Waypoint 1
    (0.7, 0.3, 0.75),  # Waypoint 2
    (0.9, 0.2, 1.0)    # End position
]

result = pipeline.generate_with_motion_control(
    prompt="A red sports car on a winding road",
    motion_type="object_trajectory",
    control_points=control_points
)
```

### Performance Optimization
```python
from ltx_video.utils.real_time_optimizations import RealTimeOptimizer, MemoryOptimizer

# Set up optimizations
optimizer = RealTimeOptimizer(pipeline)
optimizer.set_optimization_level("speed")
optimizer.enable_prompt_caching(True)

memory_optimizer = MemoryOptimizer(pipeline)
memory_optimizer.optimize_for_memory()

# Adaptive quality based on target FPS
optimized_kwargs = optimizer.adaptive_quality_scaling(
    target_fps=30.0,
    prompt="A dynamic action scene",
    num_frames=121
)

result = pipeline(**optimized_kwargs)
```

## 📊 Quality Presets

Choose from predefined quality presets based on your needs:

- **`real_time_preview()`**: 256x384, 33 frames, 10 steps - for instant previews
- **`fast_draft()`**: 512x768, 65 frames, 20 steps - for quick iterations
- **`balanced_quality()`**: 704x1216, 121 frames, 30 steps - recommended default
- **`high_quality()`**: 704x1216, 257 frames, 50 steps - for final output
- **`ultra_high_quality()`**: 1024x1792, 257 frames, 80 steps - maximum quality

## 🎨 Available Artistic Styles

- **photorealistic**: Ultra-realistic, natural lighting
- **anime**: Japanese animation style, vibrant colors
- **oil_painting**: Classical painting with textured brushstrokes
- **watercolor**: Soft, flowing artistic style
- **sketch**: Pencil sketch aesthetic
- **cyberpunk**: Futuristic neon aesthetic
- **vintage_film**: Retro film look with grain
- **noir**: High contrast, dramatic shadows
- **impressionist**: Loose brushwork, light effects
- **pop_art**: Bold colors, graphic elements

## 🎬 Motion Control Types

- **camera_pan**: Horizontal/vertical camera movement
- **camera_zoom**: Zoom in/out effects
- **camera_orbit**: Circular movement around subject
- **object_trajectory**: Custom object movement paths
- **dolly_zoom**: Vertigo effect (zoom + camera movement)
- **handheld_shake**: Realistic camera shake

## 🔧 Technical Requirements

### Hardware Requirements
- **Minimum**: RTX 3080 (10GB VRAM) for basic features
- **Recommended**: RTX 4090 (24GB VRAM) for all features
- **Real-time**: H100 (80GB VRAM) for real-time generation

### Software Dependencies
```bash
pip install torch>=2.1.0 torchvision torchaudio
pip install diffusers>=0.28.2 transformers>=4.47.2
pip install opencv-python librosa soundfile
pip install accelerate xformers  # For optimizations
```

## 🚀 Performance Tips

1. **Use Quality Presets**: Start with appropriate quality preset for your use case
2. **Enable Optimizations**: Use memory and speed optimizations for better performance
3. **Batch Processing**: Process multiple prompts together for efficiency
4. **Streaming Generation**: Use streaming for real-time applications
5. **Cache Embeddings**: Enable prompt caching for repeated generations

## 🔮 Future Enhancements

Planned features for future releases:
- **3D Scene Control**: Full 3D scene manipulation
- **Advanced Audio Sync**: Beat detection and rhythm matching
- **Interactive Generation**: Real-time parameter adjustment
- **Multi-Character Control**: Individual character animation control
- **Physics Simulation**: Realistic physics-based motion
- **Custom Style Training**: Easy custom style creation tools

## 📝 Contributing

We welcome contributions to enhance these features! Please see our contribution guidelines for more information.

## 📄 License

These enhanced features are released under the same license as the base LTX-Video model.