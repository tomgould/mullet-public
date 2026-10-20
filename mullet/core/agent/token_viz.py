"""Track and format per-turn LLM token usage into human-readable tables."""

class TokenUsageViz:
    """Track and format per-turn LLM token usage into human-readable tables."""

    def __init__(self):
        """Initialize an empty token usage tracker."""
        self.turns = []

    def record_turn(self, tokens_in: int, tokens_out: int, cached: bool = False):
        """Record token usage for a single turn."""
        self.turns.append({
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached": cached
        })

    def format_table(self) -> str:
        """Return a formatted table of recorded token usage."""
        if not self.turns:
            return "No token usage recorded yet."

        table_data = [["Turn", "Tokens In", "Tokens Out", "Cached"]]
        for i, turn in enumerate(self.turns):
            table_data.append([
                str(i + 1),
                str(turn["tokens_in"]),
                str(turn["tokens_out"]),
                "Yes" if turn["cached"] else "No"
            ])

        col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
        
        header = " | ".join(table_data[0][j].ljust(col_widths[j]) for j in range(len(table_data[0])))
        separator = "-+-".join("-" * width for width in col_widths)
        
        rows = [header, separator]
        for row_data in table_data[1:]:
            rows.append(" | ".join(str(row_data[j]).ljust(col_widths[j]) for j in range(len(row_data))))
        
        return "\n".join(rows)

    def total(self) -> dict:
        """Return aggregate token totals across all recorded turns."""
        total_in = sum(turn["tokens_in"] for turn in self.turns)
        total_out = sum(turn["tokens_out"] for turn in self.turns)
        total_cached_in = sum(turn["tokens_in"] for turn in self.turns if turn["cached"])
        total_cached_out = sum(turn["tokens_out"] for turn in self.turns if turn["cached"])

        return {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens_combined": total_in + total_out,
            "total_cached_tokens_in": total_cached_in,
            "total_cached_tokens_out": total_cached_out,
            "total_cached_tokens_combined": total_cached_in + total_cached_out
        }

if __name__ == '__main__':
    # Example usage:
    viz = TokenUsageViz()
    viz.record_turn(100, 50)
    viz.record_turn(200, 75, cached=True)
    viz.record_turn(150, 60)

    print("--- Per-turn breakdown ---")
    print(viz.format_table())

    print("\n--- Total aggregates ---")
    print(viz.total())
