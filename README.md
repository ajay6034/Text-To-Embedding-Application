## Embedding Converter Using Hugging Face Model
#### Overview
- This project provides a tool for generating embeddings from text extracted from various file formats (CSV, PDF, and JSON) using Hugging Face's Sentence Transformers. The application processes uploaded files, extracts and cleans their content, and computes embeddings, which can be downloaded for further use.

- It is implemented using Python, Gradio for the web interface, and Hugging Face for embedding generation.

### Features
- File Type Support: Handles CSV, PDF, and JSON files.
-  Extraction: Extracts text from uploaded files and preprocesses it.
- Embeddings: Generates embeddings using Hugging Face's sentence-transformers/all-MiniLM-L6-v2 model.
- Downloadable Output: Provides the top 10 embeddings in the interface and allows downloading the full embeddings as a .npy file.
- User-Friendly Interface: Powered by Gradio, the application offers an intuitive drag-and-drop file upload interface.

### Key Components
#### Text Extraction:

- CSV: Combines all rows and columns into a single text.
- PDF: Extracts text from all pages using PyPDF2.
- JSON: Recursively extracts text from JSON structures.

#### Text Preprocessing:

- Removes URLs and special characters.
- Tokenizes and lemmatizes text using SpaCy.
- Removes stopwords and non-alphabetic tokens.

#### Embedding Generation:

- Uses Hugging Face's transformer models.
- Generates sentence embeddings by averaging hidden states.

#### Web Interface:

- Built using Gradio, allows file upload and displays results.

#### Screenshot Of The Output:

![image](https://github.com/user-attachments/assets/05343ec4-10dd-499c-981d-811ae4b0665d)

