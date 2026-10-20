from mullet.core.knowledge.manager import KnowledgeManager

km = KnowledgeManager("mullet-dev")

queries = ["resource guard dispatch local LLM", "HUD rate limits display", "provider fallback chain", "team orchestration task board", "context compression structured summary"]

for query in queries:
    results = km.search(query, top_k=5)
    top = results[0] if results else {}
    src = top.get("source", top.get("metadata", {}).get("source", "?"))
    score = top.get("score", 0)
    print(f"\nQuery: {query}")
    print(f"  Results: {len(results)}, Top score: {score:.3f}, Source: {src}")