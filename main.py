import torch
import open_clip
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os

# Load CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Define your tags
TAGS = ["outdoor", "indoor", "night", "food", "people", "animal", "sports", "nature", "city"]

# Folder with images
IMAGE_FOLDER = "images/"

# Get all image files
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Prepare DataFrame to store results
df_results = pd.DataFrame(columns=["image_name", "predicted_tags"])

# Loop over images
for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    try:
        # Try opening the image
        image = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    # Preprocess image
    image_input = preprocess(image).unsqueeze(0)

    # Tokenize tags
    text_inputs = tokenizer(TAGS)

    # Compute embeddings and normalize
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute similarity and get top 3 tags
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = similarity[0].topk(3)

    predicted_tags = [TAGS[idx] for idx in top_labels]

    # Save results
    df_results = pd.concat(
        [df_results, pd.DataFrame([[img_name, ",".join(predicted_tags)]], columns=["image_name", "predicted_tags"])],
        ignore_index=True
    )

    print(img_name, "→", predicted_tags)

# Save predictions to CSV
df_results.to_csv("predictions.csv", index=False)
print("\n✅ All predictions saved to predictions.csv")
