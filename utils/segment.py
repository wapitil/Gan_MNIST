from PIL import Image
import os

def split_image(image_path, segment_width, segment_height, output_dir='segments'):
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    width, height = image.size
    segment_paths = []

    for top in range(0, height, segment_height):
        for left in range(0, width, segment_width):
            right = min(left + segment_width, width)
            bottom = min(top + segment_height, height)
            segment = image.crop((left, top, right, bottom))
            segment_path = os.path.join(output_dir, f'segment_{top}_{left}.png')
            segment.save(segment_path)
            segment_paths.append(segment_path)

    return segment_paths

# Example usage
if __name__ == "__main__":
    image_path = './gen_img/93000.png'
    segments = split_image(image_path, 32, 32)
    print(f"Segmented image paths: {segments}")
