from processing import extract_text, preprocess_text_generalized, get_embeddings_from_huggingface
import gradio as gr
import numpy as np

def process_file(file_path):
    try:
        # Step 1: Extract text
        extracted_text = extract_text(file_path)
        
        # Step 2: Preprocess text
        cleaned_text = preprocess_text_generalized(extracted_text)
        
        # Step 3: Generate embeddings
        embeddings = get_embeddings_from_huggingface(cleaned_text)
        
        # Step 4: Save embeddings to a temporary file
        temp_file_path = "embeddings.npy"
        np.save(temp_file_path, embeddings)
        
        # Return the top 10 embeddings and the file path for download
        top_10_embeddings = embeddings[:10].tolist()
        return f"Top 10 Embeddings: {top_10_embeddings}", temp_file_path
    except Exception as e:
        return str(e), None

# Define Gradio Interface
interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload a file (CSV, PDF, JSON)", type="filepath"),
    outputs=[
        gr.Textbox(label="Top 10 Embeddings"),
        gr.File(label="Download Full Embeddings"),
    ],
    title="Embedding Converter Using Hugging Face Model",
    description=(
        "Upload a file (CSV, PDF, or JSON) to  generate embeddings using "
        "Hugging Face models. View the top 10 embeddings and download  entire embedding file."
    ),
)

if __name__ == "__main__":
    interface.launch()
