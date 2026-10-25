# 0001. Cost comes from the provider's own responses

Status: Accepted

Context: A worker on a paid provider can spend money, and a worker's own report of its spending cannot be trusted. Counting tokens from the prompt gives the wrong answer for cached input and reasoning tokens.

Decision: Workers reach paid providers only through the key proxy (`mullet/jobs/executors/keyproxy.py`), which holds the key. `usage_meter.py` reads the `usage` object in every response, whether one JSON body or the last event of a stream. A cost the provider states is used as is (OpenRouter, asked with `usage.include`). Otherwise the tokens are priced from `ladder.prices`, cache rates included, and a model with no price uses `ladder.fallback_price` ($2 in, $8 out), set high on purpose. An attempt's recorded cost never falls below what the provider's responses say. Each job report compares the ledger with what each paid provider says was spent.

Consequences: Cost does not depend on any model's word. Unpriced models reach their cap early instead of looking free. An attempt killed mid-step keeps the cost charged so far, because each progress record carries the proxy's running cost. A response with tokens but no price is counted as unpriced.
