import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define our paths based on the Windows structure we built
RAW_PDF_DIR = os.path.join("data", "raw_pdfs")
INDEX_DIR = os.path.join("data", "index")

def extract_text_from_pdfs(pdf_folder):
    """
    Scans the folder for PDFs and extracts the text page by page.
    """
    all_text = []
    
    # Check if the folder exists and has files
    if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
        print(f"❌ Error: No PDFs found in {pdf_folder}. Did you download them yet?")
        return ""

    print(f"📄 Found PDFs in {pdf_folder}. Starting extraction...")
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            print(f"   -> Extracting {filename}...")
            
            try:
                # Open the PDF and read text from each page
                doc = fitz.open(filepath)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Simple text extraction. For highly complex tables, you might 
                    # later upgrade this to use OCR or markdown extraction.
                    text = page.get_text()
                    all_text.append(text)
                doc.close()
            except Exception as e:
                print(f"   -> ⚠️ Failed to read {filename}: {e}")
                
    return "\n\n".join(all_text)

def build_vector_database():
    """
    Chunks the extracted text and builds the FAISS index.
    """
    # 1. Get the raw text
    raw_text = extract_text_from_pdfs(RAW_PDF_DIR)
    
    if not raw_text.strip():
        print("❌ Extraction failed or PDFs were empty. Aborting indexing.")
        return

    # 2. Chunking Strategy
    # We use overlapping chunks to ensure a BIS standard code (e.g., IS 269) 
    # doesn't get separated from its description.
    print("✂️  Chunking the text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"   -> Created {len(chunks)} text chunks.")

    # 3. Create Embeddings
    print("🧠 Generating embeddings (this might take a minute on CPU)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Build and Save FAISS Index
    print("🏗️  Building FAISS index...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ Success! Vector database saved to {INDEX_DIR}")

if __name__ == "__main__":
    print("🚀 Starting BIS Document Parser...")
    build_vector_database()