# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os

from sentencepiece import SentencePieceProcessor
import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int]):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets


class HuggingFaceTokenizer(Tokenizer):
    """Wraps any HuggingFace tokenizer (AutoTokenizer).

    Works for models like Qwen3 that use vocab.json + merges.txt (GPT-2 BPE)
    rather than tiktoken or SentencePiece.

    ``path`` should be the directory containing the HF tokenizer files
    (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, etc.)
    or any HuggingFace model identifier.
    """

    # If an HF tokenizer reports a vocab this small after loading, we
    # almost certainly got a degenerate shell tokenizer (e.g. AutoTokenizer
    # silently constructing a bare GPTNeoXTokenizer when a directory lacks
    # tokenizer.json / vocab.json / merges.txt).  Such a tokenizer encodes
    # every input to the empty list, which silently corrupts training data
    # and evaluation.  We refuse to return it and fail loudly instead.
    _MIN_REASONABLE_VOCAB: int = 1024

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.hf_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        hf_vocab_len = len(self.hf_tok)
        if hf_vocab_len < self._MIN_REASONABLE_VOCAB:
            # Double-check by actually encoding a non-trivial probe string.  A
            # well-formed tokenizer must produce at least one content token.
            probe_ids = self.hf_tok.encode(
                "The quick brown fox jumps over the lazy dog.",
                add_special_tokens=False,
            )
            if len(probe_ids) == 0:
                raise RuntimeError(
                    f"HuggingFaceTokenizer loaded from {model_path!r} is "
                    f"degenerate: len(tokenizer)={hf_vocab_len}, class="
                    f"{type(self.hf_tok).__name__}, and a probe string "
                    f"encodes to an empty token list.  This usually means "
                    f"the directory lacks the tokenizer files "
                    f"(tokenizer.json / tokenizer_config.json / vocab.json / "
                    f"merges.txt).  Populate the directory with the real "
                    f"tokenizer, or pass an explicit tokenizer_path override "
                    f"(e.g. cfg.tokenizer_path=...) pointing at a valid "
                    f"tokenizer directory."
                )

        # n_words should match the model's embedding-table size, which may be
        # larger than the number of tokens the tokenizer actually uses (e.g.
        # Qwen3 pads vocab_size to a nice multiple for efficiency).
        # If the model directory contains a config.json with an explicit
        # vocab_size, prefer that; otherwise fall back to len(tokenizer).
        self.n_words: int = self._resolve_vocab_size(model_path)

        # BOS – Qwen3 has bos_token=None; fall back to eos if missing
        if self.hf_tok.bos_token_id is not None:
            self.bos_id: int = self.hf_tok.bos_token_id
        else:
            # Use eos as a "start" sentinel (common for Qwen / GPT-2 family)
            self.bos_id: int = self.hf_tok.eos_token_id
            logger.warning(
                f"HuggingFace tokenizer has no bos_token; using eos_token "
                f"(id={self.bos_id}) as bos fallback"
            )

        self.eos_id: int = self.hf_tok.eos_token_id
        self.pad_id: int = (
            self.hf_tok.pad_token_id
            if self.hf_tok.pad_token_id is not None
            else self.eos_id
        )

        logger.info(
            f"HuggingFace tokenizer loaded from {model_path} – "
            f"#words: {self.n_words}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}"
        )

    def _resolve_vocab_size(self, model_path: str) -> int:
        """Return the effective vocab size.

        Many models (e.g. Qwen3) pad their embedding table to a larger
        ``vocab_size`` than the number of tokens the tokenizer can actually
        produce.  If ``model_path`` is a local directory containing a
        ``config.json`` with an explicit ``vocab_size`` that is ≥ the
        tokenizer length, we use that so the embedding / output layers are
        sized correctly.  Otherwise we fall back to ``len(self.hf_tok)``.
        """
        import json

        cfg_path = Path(model_path) / "config.json"
        if cfg_path.is_file():
            try:
                with open(cfg_path) as f:
                    model_cfg = json.load(f)
                cfg_vocab = model_cfg.get("vocab_size")
                if cfg_vocab is not None and cfg_vocab >= len(self.hf_tok):
                    logger.info(
                        f"Using vocab_size={cfg_vocab} from {cfg_path} "
                        f"(tokenizer has {len(self.hf_tok)} tokens)"
                    )
                    return int(cfg_vocab)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Could not read {cfg_path}: {exc}")

        return len(self.hf_tok)

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        tokens = self.hf_tok.encode(s, add_special_tokens=False)
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.hf_tok.decode(tokens, skip_special_tokens=False)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        enc = self.hf_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets_map = enc["offset_mapping"]  # list of (start, end)
        substrs = [text[s:e] for s, e in offsets_map]
        offsets = [s for s, _ in offsets_map]
        return substrs, offsets


def build_tokenizer(name: str, path: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        return TikTokenTokenizer(path)
    elif name == "huggingface":
        return HuggingFaceTokenizer(path)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
