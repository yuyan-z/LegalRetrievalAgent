def _minmax(scores):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_search(query, bm25_index, embed_index, alpha=0.3, top_k=10):
    candidate_k = max(top_k * 3, 20)

    bm25_results = bm25_index.search(query, top_k=candidate_k, return_scores=True)
    embed_results = embed_index.search(query, top_k=candidate_k, return_scores=True)

    bm25_scores = _minmax([doc["_score"] for doc in bm25_results])
    embed_scores = _minmax([doc["_score"] for doc in embed_results])

    merged = {}

    for doc, score in zip(bm25_results, bm25_scores):
        key = doc["citation"]
        merged[key] = {
            "citation": doc["citation"],
            "text": doc["text"],
            "_bm25": score,
            "_embed": 0.0,
        }

    for doc, score in zip(embed_results, embed_scores):
        key = doc["citation"]
        if key not in merged:
            merged[key] = {
                "citation": doc["citation"],
                "text": doc["text"],
                "_bm25": 0.0,
                "_embed": 0.0,
            }
        merged[key]["_embed"] = score

    results = []
    for doc in merged.values():
        doc["_score"] = alpha * doc["_bm25"] + (1 - alpha) * doc["_embed"]
        results.append(doc)

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:top_k]
