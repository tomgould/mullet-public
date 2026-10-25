"""Shell completion scripts for mullet.

Usage:
  eval "$(mullet completion bash)"
  eval "$(mullet completion zsh)"
"""


def bash_completion() -> str:
    """Return the Bash completion script."""
    return '''_mullet_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local commands="budget cli clean completion config cost decisions doctor docs evaluate explain graph history hook hud hud-claude init ingest job live-test login logs mcp memory-sync models observations override plugins pool recall research review review-session reindex routing schedule search serve setup-claude share source stats watch"
    local flags="--client --provider --tier --local-only --max-iterations --model --base-url --verbose --json --tools --skills --export --speed --version --help"
    local cli_flags="--prompt --provider --mode --resume --continue --session --system-prompt --max-turns --speed --voice --metrics"
    local cli_slash="/help /clear /exit /model /provider /session /context /context add /context list /context clear /add-dir /compare /think /config /compact /undo /retry /forget /history /pin /note /todo /share /diff /branch /peek /open /git /run /grep /search /graph /recall /review /test /stats /doctor /cost /budget /tokens /vim"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands} ${flags}" -- "${cur}"))
    elif [[ "${cur}" == -* ]]; then
        COMPREPLY=($(compgen -W "${flags}" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "--tier" ]]; then
        COMPREPLY=($(compgen -W "LOCAL STANDARD ADVANCED CRITICAL" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "--provider" ]]; then
        COMPREPLY=($(compgen -W "local kimi gemini anthropic haiku sonnet opus bedrock openrouter" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "--speed" ]]; then
        COMPREPLY=($(compgen -W "economy balanced fast max" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "swarm" ]]; then
        local swarm_flags="--files --items --concurrency --dry-run --decomposer --timeout --output"
        COMPREPLY=($(compgen -W "${swarm_flags}" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "models" ]]; then
        local models_flags="--provider refresh"
        COMPREPLY=($(compgen -W "${models_flags}" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "pool" ]]; then
        local pool_cmds="list create delete audit prune tune docs health"
        COMPREPLY=($(compgen -W "${pool_cmds}" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "graph" ]]; then
        local graph_cmds="stats build export prune"
        COMPREPLY=($(compgen -W "${graph_cmds}" -- "${cur}"))
    elif [[ "${COMP_WORDS[1]}" == "cli" ]]; then
        if [[ "${cur}" == -* ]]; then
            COMPREPLY=($(compgen -W "${cli_flags}" -- "${cur}"))
        elif [[ "${cur}" == /* ]]; then
            COMPREPLY=($(compgen -W "${cli_slash}" -- "${cur}"))
        fi
    fi
}
complete -F _mullet_complete mullet'''


def zsh_completion() -> str:
    """Return the Zsh completion script."""
    return '''#compdef mullet

_mullet() {
    local -a commands flags
    commands=(
        'budget:Show current token budget and pressure'
        'clean:Remove stale HUD agent entries'
        'completion:Generate shell completion scripts'
        'config:Show active configuration'
        'cost:Show model pricing across tiers'
        'decisions:Browse architectural decisions extracted from commits/docs'
        'doctor:Check system health (providers, tools, config)'
        'docs:Library pool documentation management'
        'evaluate:Evaluate knowledge pool search quality'
        'explain:Show routing analysis for a task without running'
        'graph:Knowledge graph commands (stats, build, export, prune)'
        'history:Show recent sessions'
        'hook:Run a Mullet hook for Claude Code'
        'hud:HUD and statusline management'
        'hud-claude:Show Claude Code plugin-style HUD'
        'init:Generate mullet.yaml project config'
        'ingest:Ingest files into a knowledge pool'
        'job:Plan and run a gated job in a sandboxed clone'
        'live-test:Live provider connectivity test'
        'login:Authenticate with Personal Access Token'
        'logs:View LLM interaction logs (tail, stats, suggest)'
        'mcp:Start MCP server for Claude Code'
        'memory-sync:Sync session observations / decisions to memory'
        'models:List or refresh dynamic model registry'
        'observations:Browse session observations (tool-call history)'
        'override:Toggle Opus override window'
        'plugins:List installed plugins'
        'pool:Pool management (list, audit, prune, tune, validate)'
        'recall:Semantic search across past conversations'
        'research:Iterative research pipeline'
        'review:AI code review of recent git changes'
        'review-session:6-section session retrospective'
        'reindex:Reindex a knowledge pool'
        'routing:Show provider routing outcomes + insights'
        'schedule:Scheduled task management'
        'search:Search knowledge base'
        'serve:Start API + WebSocket + Prometheus server'
        'setup-claude:Register Mullet with Claude Code (MCP, skills, opt-in hooks)'
        'share:Export session as markdown/html/json'
        'source:Manage GitHub/web knowledge sources'
        'stats:Usage analytics (sessions, tools, routing)'
        'watch:Re-run task on file changes'
    )
    flags=(
        '--client[Trove pool for project context]:pool:'
        '--provider[Force provider]:provider:(local kimi gemini haiku sonnet opus anthropic bedrock openrouter)'
        '--tier[Force routing tier]:tier:(LOCAL STANDARD ADVANCED CRITICAL)'
        '--speed[Speed vs cost tradeoff]:mode:(economy balanced fast max)'
        '--local-only[Route all tasks to local GPU]'
        '--verbose[Show routing decisions and cost]'
        '--json[Output as JSON for scripting]'
        '--tools[List available agent tools]'
        '--skills[List available skills]'
        '--export[Export conversation to file]:file:_files'
        '--version[Show version]'
        '--help[Show help]'
    )

    _arguments -s $flags
    _describe 'command' commands
}

_mullet "$@"'''


def get_completion(shell: str) -> str:
    """Return the completion script for the given shell."""
    if shell == "bash":
        return bash_completion()
    elif shell == "zsh":
        return zsh_completion()
    else:
        return f"# Unsupported shell: {shell}\n# Supported: bash, zsh"
