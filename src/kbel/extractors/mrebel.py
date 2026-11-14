# Copyright 2026 LApIC
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any

from kbel.core.triple import Triple

from .abc import TripleExtractor

LOG = logging.getLogger(__name__)


class MREBELTripleExtractor(
    TripleExtractor,
    strategy_name="mrebel",
):
    """Triple extractor based on the multilingual mREBEL model.

    Uses Babelscape/mrebel-large to extract subject-predicate-object
    triples from text. Generation settings and output parsing follow the
    reference usage published by the authors
    (https://github.com/Babelscape/rebel and the
    Babelscape/mrebel-large model card).

    Unlike Stanford OpenIE, this extractor does not require Java or
    an external server. The model runs directly through PyTorch and
    Hugging Face Transformers.

    Note:
        The mREBEL model weights are distributed by Babelscape under
        CC BY-NC-SA 4.0 (non-commercial), independently of this
        library's license.

    Example:
        >>> extractor = TripleExtractor("mrebel")
        >>> triples = extractor.extract(
        ...     "Ada Lovelace trabalhou com Charles Babbage."
        ... )
        >>> for triple in triples:
        ...     print(triple)
    """

    DEFAULT_MODEL = "Babelscape/mrebel-large"
    DEFAULT_SOURCE_LANGUAGE = "en_XX"
    TARGET_LANGUAGE = "tp_XX"

    SUPPORTED_LANGUAGES = (
        "ar_AR", "ca_XX", "de_DE", "el_EL", "en_XX", "es_XX", "fr_XX",
        "hi_IN", "it_IT", "ja_XX", "ko_KR", "nl_XX", "pl_PL", "pt_XX",
        "ru_RU", "sv_SE", "vi_VN", "zh_CN",
    )

    _LANG_CODE = re.compile(r"\b[a-z]{2}_[A-Z]{2}\b")

    def __init__(
        self,
        strategy_name: str,
        model_name: str = DEFAULT_MODEL,
        source_language: str = DEFAULT_SOURCE_LANGUAGE,
        device: str | None = None,
        max_input_length: int = 256,
        max_output_length: int = 256,
        num_beams: int = 3,
        num_return_sequences: int | None = None,
        split_sentences: bool = True,
        batch_size: int = 8,
        graph: bool = False,
        graph_path: str = "graph.png",
    ):
        """Initializes the mREBEL extractor.

        Args:
            strategy_name: Name of the extraction strategy.
            model_name: Hugging Face model identifier.
            source_language: Source language code (e.g. pt_XX, en_XX).
            device: Device to use ('cuda', 'cpu' or None for automatic).
            max_input_length: Maximum number of input tokens.
            max_output_length: Maximum number of generated tokens.
            num_beams: Number of beams used during generation. Higher
                values yield more (but noisier) triples.
            num_return_sequences: How many of the beams are parsed and
                merged (deduplicated). Defaults to `num_beams`, as in the
                reference code; use 1 to keep only the best sequence.
            split_sentences: Run the model on each sentence separately.
                mREBEL tends to emit only the most salient relations of
                a long input, so splitting usually yields more triples.
            batch_size: Number of sentences processed per forward pass
                when `split_sentences` is enabled.
            graph: Render a graph image on every extract call.
            graph_path: Output path of the graph image.
        """
        assert strategy_name == self.strategy_name

        super().__init__(graph=graph, graph_path=graph_path)

        try:
            import torch
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "mREBEL requires PyTorch and Transformers. "
                "Install them with: pip install torch transformers sentencepiece"
            ) from e

        if max_input_length < 1 or max_output_length < 1:
            raise ValueError("Token limits must be positive integers.")

        if num_beams < 1:
            raise ValueError("num_beams must be a positive integer.")

        if num_return_sequences is None:
            num_return_sequences = num_beams
        if not 1 <= num_return_sequences <= num_beams:
            raise ValueError(
                "num_return_sequences must be between 1 and num_beams.")

        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        self._torch = torch
        self._model_name = model_name
        self._source_language = source_language
        self._max_input_length = max_input_length
        self._max_output_length = max_output_length
        self._num_beams = num_beams
        self._num_return_sequences = num_return_sequences
        self._split_sentences = split_sentences
        self._batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU is available."
            )

        self._device = torch.device(device)

        LOG.info(
            "Loading mREBEL model '%s' on %s.",
            model_name,
            self._device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.DEFAULT_SOURCE_LANGUAGE,
            tgt_lang=self.TARGET_LANGUAGE,
        )
        self._set_source_language(source_language)

        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        self._model.to(self._device)
        self._model.eval()

        self._decoder_start_token_id = (
            self._tokenizer.convert_tokens_to_ids(
                self.TARGET_LANGUAGE
            )
        )

        if (
            self._decoder_start_token_id is None
            or self._decoder_start_token_id
            == self._tokenizer.unk_token_id
        ):
            raise RuntimeError(
                f"Could not resolve target language token "
                f"{self.TARGET_LANGUAGE!r}."
            )

        LOG.info("mREBEL model successfully loaded.")

    def _set_source_language(self, language: str) -> None:
        """Make the tokenizer prefix inputs with `language`'s code token.

        Codes known to mBART-50 are set normally. Codes that only exist
        in mREBEL's vocabulary (ca_XX, el_EL) need the workaround from
        the model card, since `src_lang=` raises KeyError for them.
        """
        tok = self._tokenizer
        known = getattr(tok, "lang_code_to_id", None) or {}

        if language in known:
            tok.src_lang = language
        elif tok.convert_tokens_to_ids(language) not in (
                None, tok.unk_token_id):
            tok._src_lang = language
            tok.cur_lang_code_id = tok.convert_tokens_to_ids(language)
            tok.set_src_lang_special_tokens(language)
        else:
            raise ValueError(
                f"Unknown source language {language!r}. mREBEL supports: "
                f"{', '.join(self.SUPPORTED_LANGUAGES)}.")

        if language not in self.SUPPORTED_LANGUAGES:
            LOG.warning(
                "mREBEL was not trained on %r; results may be poor.",
                language)

    @staticmethod
    def _parse_triplets(generated_text: str) -> list[dict[str, str]]:
        """Parse mREBEL's generated sequence into typed triplets.

        mREBEL linearizes triples as::

            <triplet> subj <subj_type> obj1 <obj_type> rel1 <subj_type> obj2 <obj_type> rel2 ...

        i.e. after a relation, a new type tag starts another object of
        the same subject. Equivalent to `extract_triplets_typed` from the
        model card, except that incomplete triplets are always dropped.
        """
        text = (
            generated_text.replace("<s>", " ")
            .replace("</s>", " ")
            .replace("<pad>", " ")
        )
        text = MREBELTripleExtractor._LANG_CODE.sub(" ", text)

        triplets = []
        subject = object_ = relation = ""
        subject_type = object_type = ""
        state = None

        def save():
            if subject.strip() and object_.strip() and relation.strip():
                triplets.append({
                    "subject": subject.strip(),
                    "predicate": relation.strip(),
                    "object": object_.strip(),
                    "subject_type": subject_type,
                    "object_type": object_type,
                })

        for token in text.split():
            if token in ("<triplet>", "<relation>"):
                save()
                subject = object_ = relation = ""
                subject_type = object_type = ""
                state = "subject"

            elif token.startswith("<") and token.endswith(">"):
                if state == "subject":
                    subject_type = token[1:-1]
                    state = "object"
                elif state == "object":
                    object_type = token[1:-1]
                    state = "relation"
                elif state == "relation":
                    # Another object of the same subject.
                    save()
                    object_ = relation = ""
                    object_type = ""
                    subject_type = token[1:-1]
                    state = "object"

            elif state == "subject":
                subject += " " + token
            elif state == "object":
                object_ += " " + token
            elif state == "relation":
                relation += " " + token

        save()
        return triplets
    
    # Sentence boundary: [.!?…] followed by whitespace, unless the next
    # word starts in lowercase (e.g. "e.g. foo"); blank lines also split.
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?![a-z])|\n\s*\n")

    @classmethod
    def _split_into_sentences(cls, text: str) -> list[str]:
        """Split a text into sentences with a light regex heuristic."""
        return [
            s.strip()
            for s in cls._SENTENCE_BOUNDARY.split(text)
            if s and s.strip()
        ]

    def _generate(self, texts: list[str]) -> list[str]:
        """Run mREBEL on a batch of texts; returns all decoded beams."""
        inputs = self._tokenizer(
            texts,
            max_length=self._max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Same generation settings as the reference code.
        with self._torch.inference_mode():
            generated = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self._max_output_length,
                num_beams=self._num_beams,
                num_return_sequences=self._num_return_sequences,
                length_penalty=0,
                forced_bos_token_id=None,
                decoder_start_token_id=self._decoder_start_token_id,
            )

        return self._tokenizer.batch_decode(
            generated, skip_special_tokens=False
        )

    def _extract(self, text: str, prompt: Any = None) -> list[Triple]:
        if prompt is not None:
            LOG.debug("`prompt` has no effect in the mREBEL extractor.")

        chunks = (
            self._split_into_sentences(text)
            if self._split_sentences
            else [text]
        ) or [text]

        decoded: list[str] = []
        for i in range(0, len(chunks), self._batch_size):
            decoded.extend(self._generate(chunks[i:i + self._batch_size]))

        triples: dict[tuple[str, str, str], Triple] = {}
        for seq in decoded:
            for item in self._parse_triplets(seq):
                key = (item["subject"], item["predicate"], item["object"])
                triples.setdefault(
                    key,
                    Triple(
                        subject=item["subject"],
                        predicate=item["predicate"],
                        object=item["object"],
                    ),
                )

        LOG.info("Extracted %d unique triples from input text.", len(triples))
        return list(triples.values())
