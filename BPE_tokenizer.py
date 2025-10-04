import os
import json
import regex as re
from typing import BinaryIO, Iterator, Iterable
from multiprocessing import Pool
from collections import defaultdict



def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE (Byte Pair Encoding) tokenizer on the given input text file.

    Parameters
    ----------
    input_path : str
        Path to a UTF-8 encoded text file containing training data for the BPE tokenizer.
        Each line is considered part of the corpus.

    vocab_size : int
        The total size of the final vocabulary (must include initial byte-level tokens,
        all merged tokens produced during training, and the given special tokens).

    special_tokens : list[str]
        A list of user-defined special tokens (e.g., ["<|endoftext|>", "<pad>"]) to be 
        added to the vocabulary. These tokens do NOT participate in merge decisions.

    num_processes : int, optional (default=8)
        Number of parallel processes used during pre-tokenization. Each process handles
        a chunk of the input corpus split at special token boundaries. More processes
        generally mean faster pre-tokenization.

    Returns
    -------
    vocab : dict[int, bytes]
        A dictionary mapping token IDs (integers) to token values (in bytes). The token 
        IDs should be assigned sequentially starting from 0.

    merges : list[tuple[bytes, bytes]]
        A list of BPE merge operations, where each tuple represents two byte-level tokens 
        that were merged together. The list should be ordered by merge time (first merge first).
    """

    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    task_args = [(chunk, special_tokens, False) for chunk in chunk_list]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 3. Compute BPE merges
    merges : list[tuple[bytes, bytes]] = []
    pre_tokens_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break
            
        max_pair: tuple[bytes, bytes] = None
        max_cnt= -1
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i+1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)

            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j]=merged

            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    chunk, special_tokens, keep_special_tokens = args
    """
    Processes a chunk of text and returns byte pair frequency counts.

    Args:
        chunk (str): A chunk of text data (already decoded).
        special_tokens (list[str]): List of special tokens that should not be merged across.
        keep_special_tokens (bool): Whether to preserve special tokens as standalone tokens.

    Returns:
        pre_token_bytes (list[list[bytes]]): list of tokens, where each token is a list of bytes
    """
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"

    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        # if not segment:
        #     continue
        if keep_special_tokens and segment in special_tokens:
            # Treat the whole special token as a single token
            token_bytes = [segment.encode("utf-8")]
            pre_tokens_bytes.append(token_bytes)
        else:
            # Standard tokenization
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes


class Tokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer for encoding text into integer IDs and decoding back to text.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and optionally special tokens.
        
        Args:
            vocab: dict[int, bytes] - mapping from token ID to token bytes
            merges: list[tuple[bytes, bytes]] - ordered list of BPE merges
            special_tokens: list[str] | None - optional list of special tokens to add to vocabulary
        """
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))
        # Add special token to the vocabulary if it's not present
        idx = len(vocab)
        for special_token in self.special_tokens:
            if special_token not in vocab:
                vocab[idx]=special_token
                idx+=1
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()} 
        self.merges= merges
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: str - path to vocabulary file
            merges_filepath: str - path to merges file  
            special_tokens: list[str] | None - optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = token_str.encode("utf-8")  # store as bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: str - input text to encode
            
        Returns:
            list[int] - sequence of token IDs
        """
        token_ids = []
        pre_tokens_list= process_chunk((text, self.special_tokens, True))
        for token in pre_tokens_list:
            for merge in self.merges:
                merged_token = merge[0]+merge[1]
                new_tokens = []
                i=0
                while i<len(token):
                    if i<len(token)-1 and token[i]+token[i+1]==merged_token:
                        new_tokens.append(merged_token)
                        i+=2
                    else:
                        new_tokens.append(token[i])
                        i+=1
                token = new_tokens
            for i in range(len(token)):
                token_ids.append(self.vocab_reversed.get(token[i]))
        return token_ids

            
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        Required for memory-efficient tokenization of large files.
        
        Args:
            iterable: Iterable[str] - iterable of strings (e.g., file handle)
            
        Returns:
            Iterator[int] - generator yielding token IDs
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]  
            else:
                token = bytes(replacement_char, encoding='utf-8')  
            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 
