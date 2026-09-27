"""Download the corpus and write data/corpus/*.txt plus data/metadata.json.

Run once; the output is committed so the repository works offline and the
evaluation gold set stays pinned to a corpus that cannot drift.

    python -m scripts.fetch_corpus
    python -m scripts.fetch_corpus --only raptor --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rag.atomic import write_atomic
from scripts.html_text import html_to_text

AR5IV = "https://ar5iv.labs.arxiv.org/html/"
USER_AGENT = "rag-from-scratch/0.1 (educational project)"
MIN_CHARS = 2000  # below this, ar5iv probably served an error page


@dataclass(frozen=True)
class Source:
    doc_id: str
    kind: str          # "arxiv" | "web"
    ref: str           # arXiv id, or a full URL
    title: str
    author: str
    publish_date: str  # ISO YYYY-MM-DD


# NOTE: these arXiv ids and dates are a starting list and are NOT verified.
# Step 5 of this task prints the resolved title for each one; correct any row
# whose printed title does not match, and delete any that will not fetch.
SOURCES: list[Source] = [
    Source("rag", "arxiv", "2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "Lewis et al.", "2020-05-22"),
    Source("dpr", "arxiv", "2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering", "Karpukhin et al.", "2020-04-10"),
    Source("realm", "arxiv", "2002.08909", "REALM: Retrieval-Augmented Language Model Pre-Training", "Guu et al.", "2020-02-10"),
    Source("colbert", "arxiv", "2004.12832", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction", "Khattab and Zaharia", "2020-04-27"),
    Source("colbertv2", "arxiv", "2112.01488", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", "Santhanam et al.", "2021-12-02"),
    Source("hyde", "arxiv", "2212.10496", "Precise Zero-Shot Dense Retrieval without Relevance Labels", "Gao et al.", "2022-12-20"),
    Source("raptor", "arxiv", "2401.18059", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "Sarthi et al.", "2024-01-31"),
    Source("self_rag", "arxiv", "2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique", "Asai et al.", "2023-10-17"),
    Source("step_back", "arxiv", "2310.06117", "Take a Step Back: Evoking Reasoning via Abstraction", "Zheng et al.", "2023-10-09"),
    Source("ircot", "arxiv", "2212.10509", "Interleaving Retrieval with Chain-of-Thought Reasoning", "Trivedi et al.", "2022-12-20"),
    Source("cot", "arxiv", "2201.11903", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Wei et al.", "2022-01-28"),
    Source("least_to_most", "arxiv", "2205.10625", "Least-to-Most Prompting Enables Complex Reasoning", "Zhou et al.", "2022-05-21"),
    Source("react", "arxiv", "2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", "Yao et al.", "2022-10-06"),
    Source("rag_survey", "arxiv", "2312.10997", "Retrieval-Augmented Generation for Large Language Models: A Survey", "Gao et al.", "2023-12-18"),
    Source("beir", "arxiv", "2104.08663", "BEIR: A Heterogeneous Benchmark for Zero-shot Information Retrieval", "Thakur et al.", "2021-04-17"),
    Source("sbert", "arxiv", "1908.10084", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "Reimers and Gurevych", "2019-08-27"),
    Source("bert", "arxiv", "1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al.", "2018-10-11"),
    Source("fid", "arxiv", "2007.01282", "Leveraging Passage Retrieval with Generative Models", "Izacard and Grave", "2020-07-02"),
    Source("ragas", "arxiv", "2309.15217", "RAGAS: Automated Evaluation of Retrieval Augmented Generation", "Es et al.", "2023-09-26"),
    Source("crag", "arxiv", "2401.15884", "Corrective Retrieval Augmented Generation", "Yan et al.", "2024-01-29"),
    Source("query_rewriting", "arxiv", "2305.14283", "Query Rewriting for Retrieval-Augmented Large Language Models", "Ma et al.", "2023-05-23"),
    Source("ir_llm_survey", "arxiv", "2308.07107", "Large Language Models for Information Retrieval: A Survey", "Zhu et al.", "2023-08-14"),
    Source("splade", "arxiv", "2107.05720", "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking", "Formal et al.", "2021-07-12"),
    Source("contriever", "arxiv", "2112.09118", "Unsupervised Dense Information Retrieval with Contrastive Learning", "Izacard et al.", "2021-12-16"),
    Source("atlas", "arxiv", "2208.03299", "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "Izacard et al.", "2022-08-05"),
    Source("replug", "arxiv", "2301.12652", "REPLUG: Retrieval-Augmented Black-Box Language Models", "Shi et al.", "2023-01-30"),
    Source("lost_in_middle", "arxiv", "2307.03172", "Lost in the Middle: How Language Models Use Long Contexts", "Liu et al.", "2023-07-06"),
    Source("hotpotqa", "arxiv", "1809.09600", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", "Yang et al.", "2018-09-25"),
    Source("natural_questions", "arxiv", "1906.00300", "Latent Retrieval for Weakly Supervised Open Domain Question Answering", "Lee et al.", "2019-06-01"),
    Source("rankgpt", "arxiv", "2304.09542", "Is ChatGPT Good at Search? Investigating LLMs as Re-Ranking Agents", "Sun et al.", "2023-04-19"),
    Source("multi_vector", "arxiv", "2005.00181", "Sparse, Dense, and Attentional Representations for Text Retrieval", "Luan et al.", "2020-05-01"),
    Source("prompt_survey", "arxiv", "2107.13586", "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods", "Liu et al.", "2021-07-28"),
    Source("instructor", "arxiv", "2212.09741", "One Embedder, Any Task: Instruction-Finetuned Text Embeddings", "Su et al.", "2022-12-19"),
    Source("gtr", "arxiv", "2112.07899", "Large Dual Encoders Are Generalizable Retrievers", "Ni et al.", "2021-12-15"),
    Source("mteb", "arxiv", "2210.07316", "MTEB: Massive Text Embedding Benchmark", "Muennighoff et al.", "2022-10-13"),
    Source("longformer", "arxiv", "2004.05150", "Longformer: The Long-Document Transformer", "Beltagy et al.", "2020-04-10"),
    Source("t5", "arxiv", "1910.10683", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "Raffel et al.", "2019-10-23"),
    Source("attention", "arxiv", "1706.03762", "Attention Is All You Need", "Vaswani et al.", "2017-06-12"),
]


def source_url(source: Source) -> str:
    if source.kind == "arxiv":
        return f"{AR5IV}{source.ref}"
    if source.kind == "web":
        return source.ref
    raise ValueError(f"unknown source kind: {source.kind}")


def metadata_record(source: Source) -> dict[str, str]:
    return {
        "title": source.title,
        "source": source.kind,
        "publish_date": source.publish_date,
        "author": source.author,
        "url": source_url(source),
    }


def fetch_text(url: str, opener=urllib.request.urlopen) -> str:
    """Fetch a URL and return its visible text."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=60) as response:
        raw = response.read()
    return html_to_text(raw.decode("utf-8", errors="replace"))


def save_metadata(path: Path, documents: dict[str, dict[str, str]]) -> None:
    """Write metadata.json atomically: temporary file, then rename.

    Called after every document that lands on disk (see main()) so that an
    interruption at any point - Ctrl+C, a crash, a killed process - leaves
    this file consistent with whatever .txt files already exist, rather than
    only being correct once an entire run finishes.

    Writing in place would undercut that: a kill mid-write leaves truncated
    JSON, and the next run then fails on a parse error instead of on a
    missing entry.
    """
    write_atomic(
        path,
        lambda target: target.write_text(
            json.dumps({"documents": documents}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.json"))
    parser.add_argument("--only", action="append", help="fetch only these doc_ids")
    parser.add_argument("--force", action="store_true", help="refetch existing files")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args(argv)

    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SOURCES if not args.only or s.doc_id in args.only]

    documents: dict[str, dict[str, str]] = {}
    if args.metadata.is_file():
        documents = json.loads(args.metadata.read_text(encoding="utf-8"))["documents"]

    failures: list[str] = []
    for source in selected:
        target = args.corpus_dir / f"{source.doc_id}.txt"
        if target.is_file() and not args.force:
            print(f"skip   {source.doc_id} (exists)")
            documents[source.doc_id] = metadata_record(source)
            save_metadata(args.metadata, documents)
            continue
        try:
            text = fetch_text(source_url(source))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"FAIL   {source.doc_id}: {exc}", file=sys.stderr)
            failures.append(source.doc_id)
            continue
        if len(text) < MIN_CHARS:
            print(f"FAIL   {source.doc_id}: only {len(text)} chars", file=sys.stderr)
            failures.append(source.doc_id)
            continue
        target.write_text(text, encoding="utf-8")
        documents[source.doc_id] = metadata_record(source)
        save_metadata(args.metadata, documents)
        print(f"ok     {source.doc_id}  {len(text):>7} chars  {source.title[:60]}")
        time.sleep(args.delay)

    print(f"\n{len(documents)} documents in {args.metadata}")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
