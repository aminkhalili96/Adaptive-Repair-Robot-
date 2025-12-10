"""
Video recording utilities for PyBullet simulation.

Captures frames during simulation and compiles them into an MP4 video.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoRecorder:
    """
    Records simulation frames and exports them as an MP4 video.
    
    Usage:
        recorder = VideoRecorder(width=640, height=480)
        recorder.start()
        
        # During simulation loop:
        for step in range(num_steps):
            p.stepSimulation()
            rgb_frame = env.capture_image()[0]  # Get RGB from environment
            recorder.add_frame(rgb_frame)
        
        recorder.save("output.mp4")
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        output_dir: str = "temp"
    ):
        """
        Initialize the video recorder.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second for output video
            output_dir: Directory to save output videos
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.frames: List[np.ndarray] = []
        self.is_recording = False
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def start(self) -> None:
        """Start recording (clears any previous frames)."""
        self.frames = []
        self.is_recording = True
        
    def stop(self) -> None:
        """Stop recording."""
        self.is_recording = False
        
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the recording.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
        """
        if not self.is_recording:
            return
            
        # Ensure frame is the correct shape
        if frame.shape[:2] != (self.height, self.width):
            if HAS_CV2:
                frame = cv2.resize(frame, (self.width, self.height))
            else:
                # Simple resize by slicing/padding
                h, w = frame.shape[:2]
                if h > self.height:
                    frame = frame[:self.height, :]
                if w > self.width:
                    frame = frame[:, :self.width]
                    
        self.frames.append(frame.copy())
        
    def save(
        self,
        filename: str = "simulation_result.mp4",
        codec: str = "mp4v"
    ) -> Optional[str]:
        """
        Save recorded frames as an MP4 video.
        
        Args:
            filename: Output filename (placed in output_dir)
            codec: FourCC codec (default: mp4v for compatibility)
            
        Returns:
            Full path to saved video, or None if save failed
        """
        if not self.frames:
            print("No frames to save")
            return None
            
        if not HAS_CV2:
            print("OpenCV (cv2) is required for video export")
            return None
            
        # Full output path
        output_path = os.path.join(self.output_dir, filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not writer.isOpened():
            print(f"Failed to create video writer for {output_path}")
            return None
            
        # Write frames (convert RGB to BGR for OpenCV)
        for frame in self.frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
            
        writer.release()
        print(f"Video saved to {output_path} ({len(self.frames)} frames)")
        
        return output_path
        
    def get_frame_count(self) -> int:
        """Get the number of recorded frames."""
        return len(self.frames)
        
    def clear(self) -> None:
        """Clear all recorded frames."""
        self.frames = []
        self.is_recording = False


def run_simulation_with_recording(
    env,
    num_steps: int = 240,
    output_filename: str = "simulation_result.mp4",
    step_callback: Optional[Callable] = None,
    frame_skip: int = 4
) -> Optional[str]:
    """
    Run a simulation and record it to video.
    
    Args:
        env: SimulationEnvironment instance (should be headless)
        num_steps: Number of simulation steps to run
        output_filename: Output video filename
        step_callback: Optional callback called each step with (step_num, env)
        frame_skip: Record every Nth frame (reduces file size)
        
    Returns:
        Path to saved video file, or None if failed
    """
    recorder = VideoRecorder(
        width=env.camera_width,
        height=env.camera_height,
        fps=30 // frame_skip  # Adjust FPS based on frame skip
    )
    
    recorder.start()
    
    for step in range(num_steps):
        # Step simulation
        env.step()
        
        # Capture frame (with skip)
        if step % frame_skip == 0:
            rgb, _, _ = env.capture_image()
            recorder.add_frame(rgb)
            
        # Optional callback for custom logic
        if step_callback:
            step_callback(step, env)
            
    recorder.stop()
    
    return recorder.save(output_filename)
