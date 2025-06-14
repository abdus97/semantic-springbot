import os
import glob
import argparse
import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embedding function that matches ChromaDB's expected interface."""
        return self.model.encode(input, show_progress_bar=False).tolist()

class CodebaseIngestor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "./vector_store"):
        """Initialize the codebase ingestor with embedding model and database."""
        self.model_name = model_name
        self.db_path = db_path
        self.embedding_function = None
        self.chroma_client = None
        self.collection = None
        
    def initialize(self):
        """Initialize SentenceTransformer model and ChromaDB client."""
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.embedding_function = SentenceTransformerEmbeddingFunction(self.model_name)
            
            logger.info(f"Initializing ChromaDB client at: {self.db_path}")
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def setup_collection(self, collection_name: str = "springboot-codebase"):
        """Create or load ChromaDB collection."""
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if collection_name in existing_collections:
                logger.info(f"Loading existing collection: {collection_name}")
                self.collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                # Get current count
                current_count = self.collection.count()
                logger.info(f"Collection already contains {current_count} documents")
            else:
                logger.info(f"Creating new collection: {collection_name}")
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    def get_files(self, base_path: str, extensions: List[str]) -> List[str]:
        """Get all files with specified extensions from the base path."""
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Path does not exist: {base_path}")
            
        files = []
        for ext in extensions:
            pattern = f"{base_path}/**/*.{ext}"
            found_files = glob.glob(pattern, recursive=True)
            files.extend(found_files)
            
        # Remove duplicates and sort
        files = list(set(files))
        files.sort()
        
        logger.info(f"Found {len(files)} files with extensions: {extensions}")
        return files

    def chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into smaller pieces with optional overlap.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (approximate word count)
            overlap: Number of words to overlap between chunks
        """
        if not text or not text.strip():
            return []
            
        words = text.split()
        if len(words) <= max_tokens:
            return [text]
            
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + max_tokens, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Move start index, accounting for overlap
            if end_idx >= len(words):
                break
            start_idx = end_idx - overlap
            
        return chunks

    def should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped based on common patterns."""
        skip_patterns = [
            '.git', 'node_modules', 'target', 'build', '.gradle',
            '__pycache__', '.vscode', '.idea', 'logs'
        ]
        
        file_path_lower = file_path.lower()
        return any(pattern in file_path_lower for pattern in skip_patterns)

    def read_file_safely(self, file_path: str) -> Tuple[str, bool]:
        """
        Safely read file content with multiple encoding attempts.
        
        Returns:
            Tuple of (content, success_flag)
        """
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content, True
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                return "", False
                
        logger.warning(f"Could not read file with any encoding: {file_path}")
        return "", False

    def ingest_codebase(self, path: str, extensions: List[str] = None, batch_size: int = 100):
        """
        Ingest codebase into vector database.
        
        Args:
            path: Path to codebase
            extensions: List of file extensions to include
            batch_size: Number of documents to process in each batch
        """
        if extensions is None:
            extensions = ["java", "py", "js", "ts", "yml", "yaml", "md", "txt", "properties", "xml"]
            
        try:
            files = self.get_files(path, extensions)
            if not files:
                logger.warning("No files found to ingest")
                return
                
            total_chunks = 0
            failed_files = 0
            batch_documents = []
            batch_metadatas = []
            batch_ids = []
            
            for file_idx, file_path in enumerate(files, 1):
                if self.should_skip_file(file_path):
                    logger.debug(f"Skipping file: {file_path}")
                    continue
                    
                logger.info(f"Processing file {file_idx}/{len(files)}: {file_path}")
                
                content, success = self.read_file_safely(file_path)
                if not success:
                    failed_files += 1
                    continue
                    
                if not content.strip():
                    logger.debug(f"Skipping empty file: {file_path}")
                    continue
                    
                chunks = self.chunk_text(content)
                relative_path = os.path.relpath(file_path, path)
                
                for chunk_idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                        
                    doc_id = f"{relative_path}_chunk_{chunk_idx}"
                    metadata = {
                        "file_path": relative_path,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "file_extension": os.path.splitext(file_path)[1][1:],  # Remove the dot
                        "file_size": len(content)
                    }
                    
                    batch_documents.append(chunk)
                    batch_metadatas.append(metadata)
                    batch_ids.append(doc_id)
                    total_chunks += 1
                    
                    # Process batch when it reaches batch_size
                    if len(batch_documents) >= batch_size:
                        self._add_batch_to_collection(batch_documents, batch_metadatas, batch_ids)
                        batch_documents.clear()
                        batch_metadatas.clear()
                        batch_ids.clear()
            
            # Process remaining documents in the last batch
            if batch_documents:
                self._add_batch_to_collection(batch_documents, batch_metadatas, batch_ids)
            
            logger.info(f"Successfully ingested {total_chunks} chunks from {len(files) - failed_files} files")
            if failed_files > 0:
                logger.warning(f"Failed to process {failed_files} files")
                
        except Exception as e:
            logger.error(f"Failed to ingest codebase: {e}")
            raise

    def _add_batch_to_collection(self, documents: List[str], metadatas: List[dict], ids: List[str]):
        """Add a batch of documents to the collection."""
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.debug(f"Added batch of {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add batch to collection: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Ingest codebase into vector database")
    parser.add_argument("--path", required=True, help="Path to your codebase")
    parser.add_argument("--extensions", nargs="+", 
                       help="File extensions to include (default: java py js ts yml yaml md txt properties xml)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="SentenceTransformer model name")
    parser.add_argument("--db-path", default="./vector_store", 
                       help="Path to vector database")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing documents")
    parser.add_argument("--collection-name", default="springboot-codebase",
                       help="Name of the ChromaDB collection")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what files would be processed without actually ingesting")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Starting ingestion from path: {args.path}")
        logger.info(f"Target collection: {args.collection_name}")
        logger.info(f"Database path: {args.db_path}")
        
        # Validate path exists
        if not os.path.exists(args.path):
            logger.error(f"Path does not exist: {args.path}")
            return 1
        
        ingestor = CodebaseIngestor(model_name=args.model, db_path=args.db_path)
        ingestor.initialize()
        ingestor.setup_collection(args.collection_name)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be ingested")
            extensions = args.extensions or ["java", "py", "js", "ts", "yml", "yaml", "md", "txt", "properties", "xml"]
            files = ingestor.get_files(args.path, extensions)
            logger.info(f"Would process {len(files)} files:")
            for file_path in files[:10]:  # Show first 10 files
                logger.info(f"  - {file_path}")
            if len(files) > 10:
                logger.info(f"  ... and {len(files) - 10} more files")
        else:
            ingestor.ingest_codebase(args.path, args.extensions, args.batch_size)
            logger.info("✅ Ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        logger.error(f"Full error trace: {traceback.format_exc()}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())