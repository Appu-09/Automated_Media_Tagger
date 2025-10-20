# Automated Media Tagger

A Python project that automatically tags images using AI. This tool leverages OpenAI's CLIP model to predict descriptive labels for images and can evaluate predictions against manual tags.

 Project Overview

The Automated Media Tagger:

- Accepts images as input.
- Uses a **pre-trained CLIP model** to extract features and predict relevant tags (like `outdoor`, `food`, `people`, `sports`, etc.).
- Can compare predicted tags with a manually labeled CSV to calculate **accuracy, extra tags, and missing tags**.
- Includes a **Gradio web interface** for easy interaction:
- Single image tagging
- Batch evaluation (CSV + image folder)

**Benefits**:

- Speeds up image labeling for machine learning datasets.
- Ensures more consistent and accurate tagging compared to manual labeling.
- Recruiters or teams can quickly evaluate media content without manual effort.

## 🔹 Technologies Used

Python  
PyTorch – for deep learning model inference  
OpenAI CLIP – vision-language model  
Pandas – CSV and data handling  
Gradio – easy-to-use web interface  
PIL (Pillow) – image preprocessing

Future Improvements: 

Integrate RAG (Retrieval-Augmented Generation) for context-based tagging.

Use larger custom datasets to fine-tune CLIP for higher accuracy.

Improve batch processing efficiency for large image datasets.

Deploy as a web application on Hugging Face Spaces.

Deploy as a web application on Hugging Face Spaces
cd Automated_Media_Tagger

