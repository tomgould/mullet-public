"""`mullet logs` CLI — tail / stats / suggest from the LLM-call history SQLite DB."""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone


def _get_db_path() -> Path:
    """Get the path to the interactions database."""
    from mullet.core.logging.interaction_log import default_db_path
    return default_db_path()


def _format_time(ts: str) -> str:
    """Format ISO8601 timestamp to HH:MM:SS, falling back to the raw prefix."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError, AttributeError):
        return ts[:8] if isinstance(ts, str) else "??:??:??"


def _format_latency(ms: int) -> str:
    """Format latency in ms or s."""
    if ms > 999:
        return f"{ms/1000:.1f}s"
    return f"{ms}ms"


def run_logs_tail(n: int = 20) -> None:
    """Show last n interactions in table format."""
    db_path = _get_db_path()
    
    if not db_path.exists():
        print("No interactions logged yet.")
        return
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, ts, provider, model, prompt_tokens, completion_tokens, 
                   latency_ms, success, prompt_preview
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
        """, (n,))
        
        rows = cursor.fetchall()
        if not rows:
            print("No interactions found.")
            return
            
        # Print header
        print(f"{'ID':>4} | {'Time':>8} | {'Provider':>10} | {'Model':>15} | "
              f"{'Tokens':>6} | {'Latency':>7} | {'Status':>4} | {'Preview'}")
        print("-" * 80)
        
        for row in rows:
            id_, ts, provider, model, prompt_tokens, completion_tokens, latency, success, preview = row
            tokens = prompt_tokens + completion_tokens
            status = "ok" if success else "FAIL"
            preview_text = (preview or "")[:60]
            if len(preview_text) == 60:
                preview_text += "..."
                
            print(f"{id_:>4} | {_format_time(ts):>8} | {provider:>10} | {model:>15} | "
                  f"{tokens:>6} | {_format_latency(latency):>7} | {status:>4} | {preview_text}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error reading database: {e}")


def run_logs_stats() -> None:
    """Show aggregate stats by provider and model."""
    db_path = _get_db_path()
    
    if not db_path.exists():
        print("No interactions logged yet.")
        return
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT provider, model, 
                   COUNT(*) as calls,
                   SUM(success) as successes,
                   AVG(latency_ms) as avg_latency,
                   SUM(prompt_tokens) as total_prompt,
                   SUM(completion_tokens) as total_completion
            FROM interactions
            GROUP BY provider, model
            ORDER BY provider, model
        """)
        
        rows = cursor.fetchall()
        if not rows:
            print("No interactions found.")
            return
            
        # Print header
        print(f"{'Provider':>10} | {'Model':>15} | {'Calls':>5} | {'Success%':>7} | "
              f"{'Avg(ms)':>7} | {'Prompt':>8} | {'Completion':>10}")
        print("-" * 70)
        
        total_calls = 0
        total_successes = 0
        total_prompt = 0
        total_completion = 0
        
        for row in rows:
            provider, model, calls, successes, avg_latency, prompt_tokens, completion_tokens = row
            success_rate = (successes / calls * 100) if calls > 0 else 0
            
            print(f"{provider:>10} | {model:>15} | {calls:>5} | {success_rate:>6.1f}% | "
                  f"{avg_latency:>7.0f} | {prompt_tokens:>8} | {completion_tokens:>10}")
            
            total_calls += calls
            total_successes += successes
            total_prompt += prompt_tokens
            total_completion += completion_tokens
        
        # Grand total
        if total_calls > 0:
            print("-" * 70)
            overall_success = (total_successes / total_calls * 100)
            print(f"{'TOTAL':>10} | {'':>15} | {total_calls:>5} | {overall_success:>6.1f}% | "
                  f"{'':>7} | {total_prompt:>8} | {total_completion:>10}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error reading database: {e}")


def run_logs_suggest() -> None:
    """Analyze patterns and suggest better models."""
    db_path = _get_db_path()
    
    if not db_path.exists():
        print("No interactions logged yet.")
        return
    
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Get data from last 7 days
        seven_days_ago = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)).isoformat()
        
        cursor.execute("""
            SELECT provider, model, 
                   COUNT(*) as calls,
                   SUM(success) as successes,
                   AVG(latency_ms) as avg_latency
            FROM interactions
            WHERE ts >= ?
            GROUP BY provider, model
            HAVING calls >= 5
            ORDER BY provider, model
        """, (seven_days_ago,))
        
        rows = cursor.fetchall()
        if not rows:
            print("Not enough data in last 7 days for suggestions.")
            return
            
        issues_found = False
        
        for row in rows:
            provider, model, calls, successes, avg_latency = row
            failure_rate = ((calls - successes) / calls * 100) if calls > 0 else 0
            
            if failure_rate > 5:
                print(f"{provider}/{model}: {failure_rate:.1f}% failure rate → consider switching providers")
                issues_found = True
                
            if avg_latency > 30000:  # 30 seconds
                print(f"{provider}/{model}: avg {avg_latency/1000:.1f}s latency → consider faster providers")
                issues_found = True
        
        if not issues_found:
            print("All providers performing well.")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error reading database: {e}")
