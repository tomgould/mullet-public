# 0004. `mullet serve` needs a token and refuses foreign Host and Origin

Status: Accepted

Context: A web page in the user's browser can send requests to `localhost` as easily as to its own site. CORS headers stop the page reading the answer, not the server acting on the request. DNS rebinding makes a hostile name resolve to the local server.

Decision: `mullet/server/guard.py` checks every request before anything else runs. The `Host` must be `localhost`, `127.0.0.1`, `::1`, the bound address, or in `server.allowed_hosts`. An `Origin`, when present, must match `server.allowed_origins` (default `http://localhost:*` and `http://127.0.0.1:*`). Both fail with 403. In solo mode every path except `/health` needs the install's token, or the request gets 401. The token is made on first use and kept in `serve-token` in the config directory, mode 0600. It is sent as `Authorization: Bearer` or `X-Mullet-Token`, and a WebSocket may use `?token=`. `mullet serve --show-token` prints it.

Consequences: Other web pages and other machines cannot drive the server. Clients must send the token. Binding to a non-local address needs `server.allowed_hosts`. `server.require_token: false` turns the token check off. Enterprise mode uses personal access tokens instead.
