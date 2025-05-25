# search_images.py
import torch
import numpy as np
import os
import shutil # Import shutil for file operations
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity # Cosine similarity is a common metric for embeddings

def search_images(text_query, embeddings_file="image_embeddings.npz", top_n=5):
    """
    Searches for images based on a text query using saved embeddings.

    Args:
        text_query (str): The text prompt to search for.
        embeddings_file (str): Path to the saved embeddings file (.npz).
        top_n (int): Number of top results to return.

    Returns:
        list: A list of tuples, where each tuple contains (image_path, similarity_score).
    """
    try:
        # Load the saved embeddings and paths
        data = np.load(embeddings_file)
        image_embeddings = data['embeddings']
        image_paths = data['paths']
        print(f"Loaded {len(image_paths)} image embeddings from {embeddings_file}")

    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeddings_file}")
        print("Please run index_images.py first to generate the embeddings.")
        return []
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []

    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Process and encode the text query
    inputs = processor(text=text_query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Get the text features (embeddings)
        text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # Normalize the embedding
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity between the text embedding and all image embeddings
    # Reshape text_embedding for cosine_similarity which expects 2D arrays
    similarities = cosine_similarity(text_embedding.numpy(), image_embeddings)[0]

    # Get the indices of the top N most similar images
    # Use argsort and slice the last top_n elements, then reverse for descending order
    top_n_indices = np.argsort(similarities)[-top_n:][::-1]

    # Get the corresponding image paths and scores
    results = [(image_paths[i], similarities[i]) for i in top_n_indices]

    return results


if __name__ == "__main__":
    # Define the results folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, "search_results")

    # Clear the results folder if it exists
    if os.path.exists(results_folder):
        print(f"Clearing previous results in {results_folder}...")
        shutil.rmtree(results_folder)

    # Create a new results folder
    os.makedirs(results_folder)
    print(f"Created new results folder: {results_folder}")

    # Example Usage:
    search_query = input("Enter your text query for image search: ")
    search_results = search_images(search_query)

    if search_results:
        print(f"\nTop results for query '{search_query}':")
        for i, (path, score) in enumerate(search_results):
            print(f"  {i+1}: Score: {score:.4f} - {path}")
            # Copy the image to the results folder
            try:
                shutil.copy2(path, results_folder) # Use copy2 to preserve metadata
                print(f"  Copied {os.path.basename(path)} to {results_folder}")
            except Exception as e:
                print(f"  Could not copy {os.path.basename(path)}: {e}")

        print(f"\nSearch complete. Images copied to {results_folder}")
    else:
        print("\nNo results found.")