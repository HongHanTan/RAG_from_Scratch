"""Multi-representation indexing.

RAG.pdf's argument: a chunk is a poor search key for a document, because it
describes only its own paragraph. Summarising the whole document gives a key
that describes the whole document — and once a summary matches, the thing
worth handing to the generator is the document, not the summary.

So the index holds one node per document containing its summary, and a
separate docstore holds the full text. Retrieval matches summaries;
`expand_to_documents` swaps in the documents afterwards.
"""

from __future__ import annotations

from rag.chunking import RetrievedChunk, make_summary_chunk
from rag.store import VectorStore
from rag.summarise import DOC_SUMMARY_TEMPLATE, SummaryError, summarise


def build_multi_representation(documents, embedder, llm, trace=None) -> VectorStore:
    """One summary node per document, with the full text in the docstore.

    A document whose summary fails is skipped and recorded rather than
    aborting the build: losing one document from the index is much better
    than losing the index. If every document fails, that is not a degraded
    index but no index, and it raises.
    """
    nodes = []
    summaries = []
    docstore = {}
    doc_meta = {}

    for doc in documents:
        try:
            summary = summarise(doc.text, llm, DOC_SUMMARY_TEMPLATE)
        except SummaryError as exc:
            if trace is not None:
                trace.degraded(
                    f"summary failed for {doc.doc_id}: {exc}",
                    "skipping that document",
                )
            continue
        nodes.append(
            make_summary_chunk(
                chunk_id=f"{doc.doc_id}:summary",
                doc_id=doc.doc_id,
                index=0,
                text=summary,
                level=1,
            )
        )
        summaries.append(summary)
        docstore[doc.doc_id] = doc.text
        doc_meta[doc.doc_id] = {
            "title": doc.title,
            "source": doc.source,
            "topic": doc.topic,
            "publish_date": doc.publish_date,
            "author": doc.author,
            "url": doc.url,
        }

    if not nodes:
        raise SummaryError(
            "no documents could be summarised; refusing to build an empty index"
        )

    store = VectorStore(vectors=embedder.encode(summaries), chunks=nodes)
    store.docstore = docstore
    store.doc_meta = doc_meta
    return store


def expand_to_documents(
    retrieved: list[RetrievedChunk], docstore: dict
) -> list[RetrievedChunk]:
    """Swap each retrieved summary for the document it summarises.

    A hit whose document is missing from the docstore is left as the summary
    rather than dropped — a slightly worse context beats a hole in the
    results.
    """
    from dataclasses import replace

    expanded = []
    for item in retrieved:
        text = docstore.get(item.chunk.doc_id)
        if text is None:
            expanded.append(item)
            continue
        expanded.append(replace(item, chunk=replace(item.chunk, text=text)))
    return expanded
