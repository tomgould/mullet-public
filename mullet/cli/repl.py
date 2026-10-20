"""Interactive REPL mode for Mullet.

Conversation history preserved across turns. Supports slash commands
for skill invocation and Ctrl+C for interrupting agent runs.
"""

import signal
import sys
import logging

from mullet.core.agent.runner import AgentRunner
from mullet.core.providers.base import Provider
from mullet.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_repl(
    provider: Provider,
    tool_registry: ToolRegistry,
    toolsets: list[str],
    system_prompt: str = "",
    max_iterations: int = 90,
    verbose: bool = False,
    allowed_providers: frozenset[str] | None = None,
):
    """Run an interactive REPL session."""
    conversation_history: list[dict] = []
    current_agent: AgentRunner | None = None

    # Initialize session for persistence
    from mullet.core.agent.session import Session
    current_session = Session()
    current_session.metadata["provider"] = provider.name

    print("Mullet interactive mode. Type 'exit' or Ctrl+D to quit.")
    print(f"Provider: {provider.name} ({getattr(provider, 'model', 'default')})")
    print("Commands: /skill-name, exit, clear")
    print()

    # Load skills for autocomplete/matching
    skill_registry = None
    try:
        from mullet.core.skills.loader import SkillLoader
        from mullet.core.skills.registry import SkillRegistry
        loader = SkillLoader()
        skill_registry = SkillRegistry(loader=loader)
        skill_registry.load_all()
        skills = skill_registry.list_skills()
        if skills:
            print(f"Skills available: {', '.join('/' + s['name'] for s in skills)}")
            print()
    except Exception as exc:
        logger.warning("repl: failed to load skills: %s", exc)

    def _interrupt_handler(signum, frame):
        if current_agent:
            current_agent.interrupt("Interrupted by user (Ctrl+C)")
            print("\n[Interrupting agent...]", file=sys.stderr)
        else:
            print("\nGoodbye.")
            sys.exit(0)

    previous_sigint = signal.signal(signal.SIGINT, _interrupt_handler)

    try:
        while True:
            try:
                user_input = input("mullet> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            if user_input.lower() in ("help", "/help"):
                print("Commands:")
                print("  /skill-name     Invoke a skill (e.g., /code-review)")
                print("  clear           Clear conversation history")
                print("  history         Show conversation history")
                print("  help            Show this help")
                print("  exit            Exit the REPL")
                if skill_registry:
                    skills = skill_registry.list_skills()
                    if skills:
                        print(f"\nSkills: {', '.join('/' + s['name'] for s in skills)}")
                continue
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("Conversation cleared.")
                continue
            if user_input.lower() == "history":
                if not conversation_history:
                    print("(no conversation history)")
                for msg in conversation_history:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if content:
                        print(f"[{role}] {content[:100]}")
                continue

            # Check for skill match
            agent_prompt = system_prompt
            if skill_registry:
                match = skill_registry.match(user_input)
                if match:
                    invocation = skill_registry.invoke(match.skill.metadata.name)
                    agent_prompt += f"\n\n{invocation.content}"
                    if verbose:
                        print(f"  [skill: {match.skill.metadata.name}]", file=sys.stderr)

            current_agent = AgentRunner(
                provider=provider,
                tool_registry=tool_registry,
                toolsets=toolsets,
                system_prompt=agent_prompt,
                max_iterations=max_iterations,
                allowed_providers=allowed_providers,
            )

            result = current_agent.run(user_input, conversation_history=conversation_history)
            current_agent = None

            print(result.final_response)
            print()

            if verbose:
                print(f"  [tools: {result.tool_calls_made}, "
                      f"iterations: {result.iterations_used}, "
                      f"tokens: {result.tokens_used.total_tokens}]",
                      file=sys.stderr)

            # Preserve full conversation including tool messages
            conversation_history.append({"role": "user", "content": user_input})
            current_session.add_message("user", user_input)

            if result.messages:
                for msg in result.messages:
                    if msg.get("role") in ("assistant", "tool") and msg not in conversation_history:
                        conversation_history.append(msg)
            else:
                conversation_history.append({"role": "assistant", "content": result.final_response})

            current_session.add_message("assistant", result.final_response)
            current_session.metadata["tool_calls"] = current_session.metadata.get("tool_calls", 0) + result.tool_calls_made

            # Persist session after each turn
            try:
                current_session.save()
            except Exception as exc:
                logger.warning("repl: failed to save session: %s", exc)
    finally:
        # Restore the previously-installed SIGINT handler so the REPL's
        # custom interrupt behaviour does not leak into long-lived hosts.
        signal.signal(signal.SIGINT, previous_sigint)
