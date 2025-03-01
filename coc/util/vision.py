import os
import subprocess
from PIL import Image

def generate_video(images, output_filename, crop_params="500:374:0:0", fps=25, duration=3.2, frame_duration=0.8, codec="libx264", pix_fmt="yuv420p"):
    """
    Generates a video from a sequence of PIL.Image.Image objects using ffmpeg.
    Args:
        images: A list of PIL.Image.Image objects.
        output_filename: The name of the output video file (e.g., 'output.mp4').
        crop_params: The cropping parameters for ffmpeg's crop filter (width:height:x:y).
        fps: The desired frames per second of the output video.
        duration: Total duration of the final video.
        frame_duration: Duration each frame is held for.
        codec: The video codec to use (e.g., 'libx264').
        pix_fmt: The pixel format (e.g., 'yuv420p').
    Returns:
        None. Executes ffmpeg via subprocess. Prints the ffmpeg command.
    Raises:
        subprocess.CalledProcessError: If ffmpeg returns a non-zero exit code.
        ValueError: If input is malformed.
    """
    if not images:
        raise ValueError("images list cannot be empty")

    # Build the ffmpeg command.
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists.
        "-r", f"1/{frame_duration}",  # Input frame rate.
        "-i", "temp_img%d.png",  # Input files pattern.
        "-vf", f"crop={crop_params}, fps={fps}, format={pix_fmt}",
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-t", str(duration),
        output_filename
    ]
    print("Executing ffmpeg command:")
    print(" ".join(command))

    try:
        # Save PIL images to temporary files.
        for i, img in enumerate(images):
            img.save(f"temp_img{i}.png")

        # Run ffmpeg.
        subprocess.run(command, check=True)
    finally:
        # Cleanup temp files, ignoring any that don't exist.
        for i in range(len(images)):
            try:
                os.remove(f"temp_img{i}.png")
            except FileNotFoundError:
                pass