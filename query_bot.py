import os
import argparse
import logging
from typing import List, Dict, Any, Optional
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

class CodeSearchBot:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "./vector_store"):
        """Initialize the code search bot."""
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
            
            logger.info(f"Connecting to ChromaDB at: {self.db_path}")
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def load_collection(self, collection_name: str = "springboot-codebase"):
        """Load ChromaDB collection."""
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if collection_name not in existing_collections:
                raise ValueError(f"Collection '{collection_name}' not found. Available collections: {existing_collections}")
                
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            doc_count = self.collection.count()
            logger.info(f"Loaded collection '{collection_name}' with {doc_count} documents")
            
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise

    def semantic_search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform semantic search on the codebase.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional filters for metadata
            
        Returns:
            Dictionary containing search results
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            logger.info(f"Searching for: '{query}' (top {n_results} results)")
            
            # Perform the search
            search_kwargs = {
                "query_texts": [query],
                "n_results": min(n_results, self.collection.count()),  # Don't exceed available docs
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filter_dict:
                search_kwargs["where"] = filter_dict
                
            results = self.collection.query(**search_kwargs)
            
            if not results['documents'][0]:
                logger.warning("No results found for the query")
                return {"documents": [], "metadatas": [], "distances": []}
                
            return {
                "documents": results['documents'][0],
                "metadatas": results['metadatas'][0],
                "distances": results['distances'][0]
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def format_results(self, results: Dict[str, Any], query: str, max_snippet_length: int = 500) -> str:
        """Format search results for display."""
        docs = results['documents']
        metas = results['metadatas']
        distances = results['distances']
        
        if not docs:
            return f"No results found for query: '{query}'"
            
        output = []
        output.append(f"🔍 Search Results for: \"{query}\"")
        output.append(f"Found {len(docs)} relevant code snippets\n")
        output.append("=" * 80)
        
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            # Format metadata
            file_path = meta.get('file_path', 'Unknown')
            chunk_index = meta.get('chunk_index', 0)
            total_chunks = meta.get('total_chunks', 1)
            file_ext = meta.get('file_extension', 'txt')
            
            # Truncate document if too long
            display_doc = doc
            if len(doc) > max_snippet_length:
                display_doc = doc[:max_snippet_length] + "..."
                
            # Similarity score (lower distance = higher similarity)
            similarity_score = 1 - dist if dist <= 1 else 0
            similarity_percentage = similarity_score * 100
            
            output.append(f"\n📄 Result {i}:")
            output.append(f"   📁 File: {file_path}")
            output.append(f"   🧩 Chunk: {chunk_index + 1}/{total_chunks}")
            output.append(f"   📊 Similarity: {similarity_percentage:.1f}%")
            output.append(f"   🏷️  Type: {file_ext.upper()} file")
            output.append(f"   📏 Distance: {dist:.4f}")
            output.append(f"\n📝 Code Snippet:")
            output.append("-" * 40)
            output.append(display_doc)
            output.append("-" * 40)
            
        return "\n".join(output)

    def search_by_file_type(self, query: str, file_extension: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within specific file types."""
        filter_dict = {"file_extension": file_extension}
        return self.semantic_search(query, n_results, filter_dict)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            total_docs = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(limit=min(100, total_docs), include=["metadatas"])
            
            if not sample_results['metadatas']:
                return {"total_documents": total_docs, "file_types": {}, "error": "No metadata available"}
            
            # Analyze file types
            file_types = {}
            for meta in sample_results['metadatas']:
                ext = meta.get('file_extension', 'unknown')
                file_types[ext] = file_types.get(ext, 0) + 1
                
            return {
                "total_documents": total_docs,
                "file_types": file_types,
                "sample_size": len(sample_results['metadatas'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def interactive_search(self):
        """Start an interactive search session."""
        print("\n🤖 Welcome to the Semantic Code Search Bot!")
        print("Type your queries to search through the codebase.")
        print("Special commands:")
        print("  - 'stats': Show collection statistics")
        print("  - 'help': Show this help message")
        print("  - 'quit' or 'exit': Exit the program")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n🔍 Enter your search query: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif query.lower() == 'stats':
                    stats = self.get_collection_stats()
                    print(f"\n📊 Collection Statistics:")
                    print(f"   Total documents: {stats.get('total_documents', 'Unknown')}")
                    if 'file_types' in stats:
                        print(f"   File types found:")
                        for ext, count in stats['file_types'].items():
                            print(f"     - .{ext}: {count} chunks")
                    continue
                    
                elif query.lower() == 'help':
                    print("\n💡 Search Tips:")
                    print("  - Use natural language: 'authentication logic'")
                    print("  - Be specific: 'REST API endpoints'")
                    print("  - Technical terms work well: 'database connection'")
                    print("  - Try variations if no results found")
                    continue
                
                # Perform search
                results = self.semantic_search(query)
                formatted_results = self.format_results(results, query)
                print(f"\n{formatted_results}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Query codebase vector database")
    parser.add_argument("--query", help="Search query (if not provided, starts interactive mode)")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--db-path", default="./vector_store", help="Path to vector database")
    parser.add_argument("--collection-name", default="springboot-codebase", help="ChromaDB collection name")
    parser.add_argument("--file-type", help="Filter by file extension (e.g., 'java', 'py')")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the search bot
        bot = CodeSearchBot(model_name=args.model, db_path=args.db_path)
        bot.initialize()
        bot.load_collection(args.collection_name)
        
        # Handle different modes
        if args.stats:
            stats = bot.get_collection_stats()
            print("📊 Collection Statistics:")
            print(f"Total documents: {stats.get('total_documents', 'Unknown')}")
            if 'file_types' in stats:
                print("File types:")
                for ext, count in stats['file_types'].items():
                    print(f"  .{ext}: {count} chunks")
            return 0
            
        elif args.interactive or not args.query:
            bot.interactive_search()
            return 0
            
        else:
            # Single query mode
            if args.file_type:
                results = bot.search_by_file_type(args.query, args.file_type, args.n_results)
            else:
                results = bot.semantic_search(args.query, args.n_results)
                
            formatted_results = bot.format_results(results, args.query)
            print(formatted_results)
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())