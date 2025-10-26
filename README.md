# Astrology Q&A RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for answering quary-related questions using a combination of vector search and large language models.

## Features

- Document ingestion from both PDF and TXT files
- Vector embeddings using SentenceTransformer
- Pinecone vector database integration
- Mistral 7B Instruct v0.1 for answer generation
- Flask web interface for user interaction

## Tech Stack

- **Backend Framework**: Flask
- **LLM**: Mistral-7B-Instruct-v0.1
- **Embedding Model**: mixedbread-ai/mxbai-embed-large-v1
- **Vector Database**: Pinecone
- **Document Processing**: pdfplumber
- **Frontend**: HTML/JavaScript

## Setup

1. Clone the repository:
```bash
git clone https://github.com/itsniharsharma/RAG-projectGenAI.git
cd RAG-projectGenAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Pinecone credentials:
```env
PINECONE_API_KEY=your_api_key
INDEX_NAME=your_index_name
```

4. Place your astrology documents in the `data/` directory (supports .pdf and .txt files)

5. Initialize the vector database:
```bash
python rag_utils.py
```

6. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application and LLM integration
- `rag_utils.py`: RAG system utilities (document loading, embedding, vector search)
- `templates/`: Frontend templates
- `data/`: Directory for source documents
- `embeddings/`: Directory for storing embeddings

## Usage

1. Start the Flask server
2. Open your browser and navigate to `http://localhost:5000`
3. Enter your astrology-related question in the input field
4. The system will:
   - Retrieve relevant context from the vector database
   - Generate a response using Mistral-7B
   - Display the answer in the web interface

## License

This project is open source and available under the MIT License.
