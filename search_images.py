import torch
import numpy as np
import os
import shutil  # Import shutil for file operations
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity is a common metric for embeddings
import re  # Import regex for date extraction
from datetime import timedelta, datetime  # Import datetime for date comparison
import spacy
from dateutil import parser

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_dates_from_text(text):
    """
    Uses spacy to extract dates from a text string.
    """
    dates = []
    doc = nlp(text)
    print(f"ent count {len(doc.ents)}")
    for ent in doc.ents:
        print(f"ent: {(ent)}, {ent.label_}")
        if ent.label_ == "DATE" or ent.label_ == "CARDINAL":
            try:
                date_obj = parser.parse(ent.text)
                dates.append(date_obj)
            except ValueError:
                # If dateutil can't parse it, skip this entity
                print(f"Could not parse date entity: {ent.text}")
                pass

    print(f"{len(dates)} dates found")
    for date in dates:
         print(f"- {date}")
    return dates


def calculate_date_similarity(query_dates, image_metadata):
    """
    Calculates a date similarity score between query dates and image metadata dates.
    Returns a score between 0 and 1.
    """
    if not query_dates or image_metadata is None:
        return 0.0

    image_dates = []
    # Extract potential dates from common EXIF tags in metadata
    date_tags = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']
    for tag in date_tags:
        if tag in image_metadata:
            date_str = str(image_metadata[tag])
            try:
                # EXIF date format is usually "YYYY:MM:DD HH:MM:SS"
                # We only care about the date part for this basic implementation
                image_date = datetime.strptime(date_str.split(' ')[0], '%Y:%m:%d')
                image_dates.append(image_date)
            except (ValueError, TypeError):
                continue  # Ignore if parsing fails

    if not image_dates:
        return 0.0

    # Compare query dates with image dates
    for q_date in query_dates:
        for i_date in image_dates:
            similarity = 0.0
            delta = abs(q_date - i_date)

            if isinstance(q_date, datetime):
                if q_date.year == i_date.year and q_date.month == i_date.month and q_date.day == i_date.day:
                    similarity = 1.0  # Exact date match
                elif delta <= timedelta(days=3):  # Within 3 days if same year
                    similarity = 0.9
                elif delta <= timedelta(days=7):  # Within 7 days if same year
                    similarity = 0.8
                elif q_date.year == i_date.year and q_date.month == i_date.month: # Same Year and Month
                    similarity = 1
                elif q_date.year == i_date.year:  # Same year
                    similarity = 0.3
                elif q_date.month == i_date.month:  #Same month (ignoring all else)
                    similarity = 0.3
                elif isinstance(q_date, tuple) and len(q_date) == 2:
                    #If query only contains month and day
                    month_name, day = q_date
                    if datetime.strptime(month_name, "%B").month == i_date.month and day == i_date.day:
                        similarity = 1.0 # Month and day match, any year
                    elif datetime.strptime(month_name, "%B").month == i_date.month and abs(day - i_date.day) <= 3:
                        similarity = 0.3 # Month match, within 3 days
                    elif datetime.strptime(month_name, "%B").month == i_date.month and abs(day - i_date.day) <= 7:
                        similarity = 0.7 # Month match, within 7 days
                    else:
                        similarity = 0 #no match
                else:
                    similarity = 0  #No significant match
        
        print(f"date similarity : {similarity}")

    return similarity


def search_images(text_query, embeddings_file="image_embeddings.npz", top_n=None, min_similarity_score=None):
    """
    Searches for images based on a text query and image metadata using saved embeddings.
    Prioritizes results based on date matches in the query and metadata.
    Allows filtering by a minimum similarity score and limiting the number of results.
    """
    if top_n is None and min_similarity_score is None:
         raise ValueError("At least one of top_n or min_similarity_score must be specified.")
    try:
        data = np.load(embeddings_file, allow_pickle=True)
        image_embeddings = data['embeddings']
        image_paths = data['paths']
        image_metadatas = data['metadatas'] if 'metadatas' in data else [None] * len(image_paths)
        print(f"Loaded {len(image_paths)} image embeddings and metadatas from {embeddings_file}")

    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {embeddings_file}")
        print("Please run index_images.py first to generate the embeddings.")
        return []
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    query_dates = extract_dates_from_text(text_query)
    if query_dates:
        print(f"Extracted dates from query: {query_dates}")

    inputs = processor(text=text_query, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    image_similarities = cosine_similarity(text_embedding.numpy(), image_embeddings)[0]

    metadata_similarities = []
    metadata_texts = [str(metadata) if metadata is not None else "" for metadata in image_metadatas]

    if any(metadata_texts):
        metadata_inputs = processor(text=metadata_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            metadata_features = model.get_text_features(input_ids=metadata_inputs['input_ids'], attention_mask=metadata_inputs['attention_mask'])
            metadata_embeddings = metadata_features / metadata_features.norm(p=2, dim=-1, keepdim=True)
        metadata_similarities = cosine_similarity(text_embedding.numpy(), metadata_embeddings.numpy())[0]
    else:
        metadata_similarities = np.zeros(len(image_paths))

    combined_scores = []
    image_weight = 0.6
    metadata_text_weight = 0.2
    date_weight_multiplier = 2.0

    for i in range(len(image_paths)):
        date_similarity = calculate_date_similarity(query_dates, image_metadatas[i])
        current_date_weight = date_weight_multiplier if query_dates and date_similarity > 0 else 0

        combined_score = (image_weight * image_similarities[i] +
                          metadata_text_weight * metadata_similarities[i] +
                          current_date_weight * date_similarity)

        combined_scores.append(combined_score)

    # Apply minimum similarity score filter
    results = [(image_paths[i], combined_scores[i]) for i in range(len(image_paths))]

    if min_similarity_score is not None:
        results = [(path, score) for path, score in results if score >= min_similarity_score]

    # Sort the results by score
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Apply top_n filter
    if top_n is not None:
        results = results[:top_n]

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, "search_results")
    print(f"Clearing previous results in {results_folder}...")
    shutil.rmtree(results_folder) if os.path.exists(results_folder) else None
    os.makedirs(results_folder, exist_ok=True)
    print(f"Created new results folder: {results_folder}")

    search_query = input("Enter your text query for image search: ")

    top_n = None
    min_similarity_score = None

    while top_n is None and min_similarity_score is None:
        try:
            top_n_str = input("Enter the number of top results to return (or leave empty for no limit): ")
            top_n = int(top_n_str) if top_n_str else None

            min_similarity_score_str = input("Enter the minimum similarity score betwwen 0 and 1 (or leave empty for no limit): ")
            min_similarity_score = float(min_similarity_score_str) if min_similarity_score_str else None

            if top_n is None and min_similarity_score is None:
                print("Please enter at least one of: top_n or min_similarity_score.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    try:
         search_results = search_images(search_query, top_n=top_n, min_similarity_score=min_similarity_score)
    except ValueError as e:
         print(e)
         search_results = []

    if search_results:
        print(f"\nTop results for query '{search_query}':")
        for i, (path, score) in enumerate(search_results):
            print(f"  {i+1}: Score: {score:.4f} - {path}")
            try:
                shutil.copy2(path, results_folder)
                print(f"  Copied {os.path.basename(path)} to {results_folder}")
            except Exception as e:
                print(f"  Could not copy {os.path.basename(path)}: {e}")

        print(f"\nSearch complete. Images copied to {results_folder}")
    else:
        print("\nNo results found.")