from PIL import Image
import numpy as np

def save_binary_image(arr, path):
    """
    Saves a binary numpy array as a grayscale image.

    Args:
        arr (np.ndarray): Binary array to save.
        path (str): Output path for image.
    """
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)

def create_animation(frames, output_file = "all_test.gif", duration = 100):
    """
    Creates a GIF animation from a list of binary frames.

    Args:
        frames (list): List of 2D binary arrays.
        output_file (str): Path to output GIF file.
        duration (int): Duration per frame in milliseconds.
    """
    pil_frames = [Image.fromarray((frame * 255).astype(np.uint8)) for frame in frames]
    pil_frames[0].save(output_file, format="GIF", append_images=pil_frames[1:],
                         save_all=True, duration=duration, loop=0)
    
    print(f"Animation saved to {output_file}")
