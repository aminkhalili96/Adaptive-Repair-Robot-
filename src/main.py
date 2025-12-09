"""
Main entry point for the robotic AI simulation.

Usage:
    python -m src.main              # Run with GUI
    python -m src.main --no-gui     # Run headless
    python -m src.main --demo       # Run demo scene
"""

import argparse
import time
import sys

from src.simulation.environment import create_environment
from src.simulation.defects import spawn_demo_defects, DefectType


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robotic AI - Agentic Repair Robot")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--demo", action="store_true", help="Run demo scene")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Robotic AI - Agentic Adaptive Repair Robot")
    print("=" * 50)
    
    # Create simulation environment
    print("\n[1/3] Setting up simulation environment...")
    env = create_environment(gui=not args.no_gui)
    print(f"  ✓ PyBullet connected (GUI: {not args.no_gui})")
    print(f"  ✓ KUKA iiwa robot loaded")
    print(f"  ✓ Workpiece placed at {env.workpiece_position}")
    
    # Spawn defects
    print("\n[2/3] Spawning defects...")
    if args.demo:
        defects = spawn_demo_defects(env.workpiece_position, env.workpiece_size)
    else:
        from src.simulation.defects import spawn_random_defects
        defects = spawn_random_defects(
            env.workpiece_position,
            env.workpiece_size,
            count=3
        )
    
    env.defect_ids = [d.id for d in defects]
    
    for d in defects:
        print(f"  ✓ {d.type.value.upper()} at ({d.position[0]:.2f}, {d.position[1]:.2f}, {d.position[2]:.2f})")
    
    # Capture test image
    print("\n[3/3] Testing camera...")
    rgb, depth, seg = env.capture_image()
    print(f"  ✓ Camera capture: {rgb.shape}")
    
    print("\n" + "=" * 50)
    print("Simulation ready!")
    print("Press Ctrl+C to exit")
    print("=" * 50)
    
    # Run simulation loop
    try:
        while True:
            try:
                env.step()
                time.sleep(1.0 / 240.0)
            except p.error as e:
                # This exception is thrown when the user closes the GUI window.
                print(f"\nPyBullet GUI closed or error occurred: {e}")
                break
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")
    finally:
        print("Closing simulation environment.")
        env.close()
        print("Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
