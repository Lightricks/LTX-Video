"""
Predefined motion presets for easy camera movements and object trajectories
"""

import torch
import math
from typing import Dict, List, Tuple, Any


class MotionPresets:
    """Collection of predefined motion patterns for video generation"""
    
    @staticmethod
    def camera_pan_left_to_right(duration_frames: int = 257, speed: float = 1.0) -> Dict[str, Any]:
        """Smooth camera pan from left to right"""
        pan_values = [speed * (i / duration_frames) * 90 - 45 for i in range(duration_frames)]
        return {
            "motion_type": "camera_pan",
            "camera_params": torch.tensor([[
                50.0,  # focal_length
                pan_values[0], 0.0, 0.0,  # pan, tilt, roll
                1.0, 0.0, 0.0, 0.0, 60.0  # zoom, x, y, z, fov
            ]])
        }
    
    @staticmethod
    def camera_zoom_in(duration_frames: int = 257, zoom_factor: float = 2.0) -> Dict[str, Any]:
        """Smooth zoom in effect"""
        zoom_values = [1.0 + (zoom_factor - 1.0) * (i / duration_frames) for i in range(duration_frames)]
        return {
            "motion_type": "camera_zoom",
            "camera_params": torch.tensor([[
                50.0, 0.0, 0.0, 0.0,  # focal_length, pan, tilt, roll
                zoom_values[0], 0.0, 0.0, 0.0, 60.0  # zoom, x, y, z, fov
            ]])
        }
    
    @staticmethod
    def camera_orbit(duration_frames: int = 257, radius: float = 5.0, height: float = 0.0) -> Dict[str, Any]:
        """Circular camera orbit around subject"""
        angles = [2 * math.pi * (i / duration_frames) for i in range(duration_frames)]
        x_pos = [radius * math.cos(angle) for angle in angles]
        z_pos = [radius * math.sin(angle) for angle in angles]
        
        return {
            "motion_type": "camera_orbit",
            "camera_params": torch.tensor([[
                50.0, 0.0, 0.0, 0.0,  # focal_length, pan, tilt, roll
                1.0, x_pos[0], height, z_pos[0], 60.0  # zoom, x, y, z, fov
            ]])
        }
    
    @staticmethod
    def object_linear_trajectory(
        start_pos: Tuple[float, float] = (0.2, 0.5),
        end_pos: Tuple[float, float] = (0.8, 0.5),
        duration_frames: int = 257
    ) -> Dict[str, Any]:
        """Linear object movement trajectory"""
        control_points = []
        for i in range(8):  # 8 control points
            t = i / 7.0
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            time = t
            control_points.append([x, y, time])
        
        return {
            "motion_type": "object_trajectory",
            "motion_control_points": torch.tensor(control_points).unsqueeze(0)
        }
    
    @staticmethod
    def object_circular_trajectory(
        center: Tuple[float, float] = (0.5, 0.5),
        radius: float = 0.2,
        duration_frames: int = 257
    ) -> Dict[str, Any]:
        """Circular object movement trajectory"""
        control_points = []
        for i in range(8):  # 8 control points
            angle = 2 * math.pi * (i / 8.0)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            time = i / 7.0
            control_points.append([x, y, time])
        
        return {
            "motion_type": "object_trajectory",
            "motion_control_points": torch.tensor(control_points).unsqueeze(0)
        }
    
    @staticmethod
    def dolly_zoom_effect(duration_frames: int = 257, intensity: float = 1.0) -> Dict[str, Any]:
        """Classic dolly zoom (Vertigo effect)"""
        # Zoom in while moving camera back to maintain subject size
        zoom_values = [1.0 + intensity * (i / duration_frames) for i in range(duration_frames)]
        z_values = [-intensity * 2.0 * (i / duration_frames) for i in range(duration_frames)]
        
        return {
            "motion_type": "camera_zoom",
            "camera_params": torch.tensor([[
                50.0, 0.0, 0.0, 0.0,  # focal_length, pan, tilt, roll
                zoom_values[0], 0.0, 0.0, z_values[0], 60.0  # zoom, x, y, z, fov
            ]])
        }
    
    @staticmethod
    def handheld_shake(duration_frames: int = 257, intensity: float = 0.5) -> Dict[str, Any]:
        """Realistic handheld camera shake"""
        import random
        random.seed(42)  # For reproducible shake
        
        shake_x = [intensity * (random.random() - 0.5) * 2 for _ in range(duration_frames)]
        shake_y = [intensity * (random.random() - 0.5) * 2 for _ in range(duration_frames)]
        shake_roll = [intensity * (random.random() - 0.5) * 5 for _ in range(duration_frames)]
        
        return {
            "motion_type": "camera_pan",
            "camera_params": torch.tensor([[
                50.0, shake_x[0], shake_y[0], shake_roll[0],  # focal_length, pan, tilt, roll
                1.0, 0.0, 0.0, 0.0, 60.0  # zoom, x, y, z, fov
            ]])
        }


class StylePresets:
    """Collection of predefined artistic styles"""
    
    AVAILABLE_STYLES = {
        "photorealistic": "Ultra-realistic, high detail, natural lighting",
        "anime": "Japanese animation style, vibrant colors, stylized features",
        "oil_painting": "Classical oil painting, textured brushstrokes, rich colors",
        "watercolor": "Soft watercolor painting, flowing colors, artistic blur",
        "sketch": "Pencil sketch style, black and white, artistic lines",
        "cyberpunk": "Futuristic cyberpunk aesthetic, neon colors, high contrast",
        "vintage_film": "Retro film look, grain, warm tones, nostalgic feel",
        "noir": "Film noir style, high contrast, dramatic shadows",
        "impressionist": "Impressionist painting style, loose brushwork, light effects",
        "pop_art": "Pop art style, bold colors, high contrast, graphic elements"
    }
    
    @staticmethod
    def get_style_description(style_name: str) -> str:
        """Get description for a style"""
        return StylePresets.AVAILABLE_STYLES.get(style_name, "Unknown style")
    
    @staticmethod
    def list_available_styles() -> List[str]:
        """List all available styles"""
        return list(StylePresets.AVAILABLE_STYLES.keys())


class AudioSyncPresets:
    """Presets for audio-synchronized video generation"""
    
    @staticmethod
    def music_video_sync(audio_features: torch.Tensor, style: str = "dynamic") -> Dict[str, Any]:
        """Generate motion synchronized with music"""
        # Analyze audio features for beat detection (simplified)
        # In a real implementation, this would use proper audio analysis
        
        if style == "dynamic":
            # Fast cuts and movements for energetic music
            return {
                "motion_type": "camera_pan",
                "enable_temporal_consistency": False,  # Allow quick cuts
                "style_name": "cyberpunk",
                "style_strength": 0.8
            }
        elif style == "smooth":
            # Smooth movements for calm music
            return {
                "motion_type": "camera_orbit",
                "enable_temporal_consistency": True,
                "style_name": "watercolor",
                "style_strength": 0.6
            }
        else:
            return {}
    
    @staticmethod
    def speech_sync(audio_features: torch.Tensor) -> Dict[str, Any]:
        """Generate video synchronized with speech"""
        return {
            "motion_type": "camera_pan",
            "enable_temporal_consistency": True,
            "style_name": "photorealistic",
            "style_strength": 1.0
        }


# Example usage functions
def create_cinematic_shot(shot_type: str, **kwargs) -> Dict[str, Any]:
    """Create cinematic camera movements"""
    
    if shot_type == "establishing":
        return MotionPresets.camera_pan_left_to_right(speed=0.5, **kwargs)
    elif shot_type == "close_up_zoom":
        return MotionPresets.camera_zoom_in(zoom_factor=3.0, **kwargs)
    elif shot_type == "orbit_reveal":
        return MotionPresets.camera_orbit(radius=3.0, **kwargs)
    elif shot_type == "dolly_zoom":
        return MotionPresets.dolly_zoom_effect(intensity=1.5, **kwargs)
    elif shot_type == "handheld":
        return MotionPresets.handheld_shake(intensity=0.3, **kwargs)
    else:
        raise ValueError(f"Unknown shot type: {shot_type}")


def create_artistic_video(style: str, motion: str = "smooth", **kwargs) -> Dict[str, Any]:
    """Create stylized video with artistic effects"""
    
    motion_params = {}
    if motion == "smooth":
        motion_params = MotionPresets.camera_orbit(**kwargs)
    elif motion == "dynamic":
        motion_params = MotionPresets.camera_pan_left_to_right(speed=2.0, **kwargs)
    elif motion == "static":
        motion_params = {}
    
    return {
        **motion_params,
        "style_name": style,
        "style_strength": 0.8,
        "enable_temporal_consistency": True
    }