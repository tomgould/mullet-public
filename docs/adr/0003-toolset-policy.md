# 0003. Toolsets for delegated models are a per-provider setting

Status: Accepted

Context: Tools such as `terminal` and `file` act on the user's machine with the user's rights. A model that reads a hostile web page or file could be talked into misusing them. Claude's own providers run under the user's login; other providers are third parties.

Decision: `delegation.toolsets` sets the toolsets each provider is offered (`mullet/core/toolset_policy.py`). A value is `all` or a list of names. Without the block, `anthropic`, `claude-cli` and `claude_max` get every toolset, and any other provider gets `web`, `browser`, `session`, `vision`, `trove` and `lsp`. `default` replaces that safe set for providers not named. `delegation.file_deny` separately lists credential locations the file and docker tools refuse, and setting it replaces the whole default list.

Consequences: A new provider is safe without any configuration. Giving a provider command or file access is a deliberate edit, for example `local: [terminal, file, web, code]`. Nothing is removed from Mullet; a toolset is only left out of the list a provider is offered. `mullet doctor` warns when a non-Claude provider has `terminal`, `file`, `code`, `docker`, `git` or `testing`.
