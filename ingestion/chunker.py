"""
Semantic and recursive text chunker.
Splits documents along natural boundaries (headings, paragraphs, sentences)
while maintaining context overlap.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from agent.config import settings


@dataclass #automatically generate init, repr, and other methods for the class
class TextChunk:# a blue print that stores chunks 
    """Represents an individual text chunk with metadata."""
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticChunker:
    """
    Splits text recursively using semantic boundaries:
    1. Markdown headers (# , ## , ### )
    2. Double line breaks (\n\n = paragraphs)
    3. Single line breaks (\n)
    4. Sentence boundaries (. , ! , ? )
    5. Spaces (words)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_chunk_size: int | None = None
    ):
        # Read defaults from agent/config.py if not provided
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size

        # Regex patterns for natural language separators
        self.separators = [
            r"\n#{1,6}\s+",      # Markdown headings
            r"\n\n+",           # Paragraphs
            r"\n",              # Lines
            r"(?<=[.!?])\s+",   # Sentence boundaries (lookbehind)
            r"\s+"              # Words
        ]

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough heuristic: ~4 characters per token in English.
        Fast and avoids external tokenizer overhead during chunking.
        """
        return max(1, len(text) // 4)

    def split_text(self, text: str, metadata: Dict[str, Any] | None = None) -> List[TextChunk]:
        """ Splits text into chunks adhering to target chunk_size and chunk_overlap. """
        if not text or not text.strip():
            return []

        base_meta = metadata or {}
        raw_paragraphs = re.split(r"\n\n+", text)
        
        chunks: List[TextChunk] = []
        current_chunk = ""
        chunk_idx = 0

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds our target chunk_size
            if len(current_chunk) + len(para) > (self.chunk_size * 4):  # approx chars
                if current_chunk:
                    chunks.append(
                        TextChunk(
                            content=current_chunk.strip(),
                            chunk_index=chunk_idx,
                            token_count=self._estimate_tokens(current_chunk),
                            metadata=base_meta.copy()
                        )
                    )
                    chunk_idx += 1
                    
                    # Create overlap by keeping the trailing characters of the previous chunk
                    overlap_chars = self.chunk_overlap * 4
                    current_chunk = current_chunk[-overlap_chars:] + " " + para
                else:
                    # Paragraph itself is huge, store it directly
                    current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        # Add the remaining buffer
        if current_chunk.strip():
            chunks.append(
                TextChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_idx,
                    token_count=self._estimate_tokens(current_chunk),
                    metadata=base_meta.copy()
                )
            )

        return chunks

