# index_images.py
import os
import torch
from PIL import Image, ExifTags # Import ExifTags
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def get_image_metadata(image_path):
    """
    Extracts EXIF metadata from an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing EXIF metadata, or None if no EXIF data is found.
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is None:
            return None

        # Decode EXIF tags
        decoded_exif_data = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            # Attempt to decode bytes to string if necessary
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = str(value) # Fallback to string representation

            decoded_exif_data[tag] = value
        return decoded_exif_data
    except Exception as e:
        print(f"Could not get EXIF data for {image_path}: {e}")
        return None


def index_images(image_folder, output_file="image_embeddings.npz"):
    """
    Encodes images in a folder into vector embeddings and saves them.

    Args:
        image_folder (str): Path to the folder containing images.
        output_file (str): Path to save the embeddings and file paths.
    """
    # Load the pre-trained CLIP model and processor
    # CLIP is suitable for this task as it understands both text and images
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_paths = []
    image_embeddings = []
    image_metadatas = [] # List to store metadata

    print(f"Indexing images from: {image_folder}")

    # Iterate through files in the folder
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        # Check if the file is an image (you might want a more robust check)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                image = Image.open(file_path).convert("RGB") # Ensure image is RGB
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    # Get the image features (embeddings)
                    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
                    # Normalize the embedding
                    image_embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                image_embeddings.append(image_embedding.squeeze().numpy())
                image_paths.append(file_path)

                # Extract and store metadata
                metadata = get_image_metadata(file_path)
                image_metadatas.append(metadata)
                print(f" Indexed: {filename}")
                if metadata:
                    print(f"  Metadata: {metadata}")
                else:
                    print(f"  No metadata found for {filename}")


            except Exception as e:
                print(f"Could not process {filename}: {e}")

    # Save the embeddings and paths
    if image_embeddings:
        # Save embeddings, paths, and metadatas
        # Convert metadatas to a format that can be saved by np.savez, e.g., a list of strings or a structured array
        # For simplicity, saving as an object array of dictionaries.
        np.savez(output_file, embeddings=np.array(image_embeddings), paths=np.array(image_paths), metadatas=np.array(image_metadatas, dtype=object))
        print(f"\nSuccessfully indexed {len(image_embeddings)} images.")
        print(f"Embeddings saved to {output_file}")
    else:
        print("\nNo images found or processed in the specified folder.")


if __name__ == "__main__":
    # Example Usage:
    # Create a dummy folder and add some placeholder files if needed for testing
    # import os
    # if not os.path.exists("my_pictures"):
    #     os.makedirs("my_pictures")
    #     # Add some dummy files or actual images here

    # Replace "my_pictures" with the actual path to your image folder
    image_folder_path = input("Enter the path to the image folder: ")
    if os.path.isdir(image_folder_path):
        index_images(image_folder_path)
    else:
        print("Error: The provided path is not a valid directory.")
