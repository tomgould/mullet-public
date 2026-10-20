# Mullet on Mac — Complete Setup Guide

> **Mullet is a smart router for AI coding assistants.** Instead of sending every question to expensive AI models, Mullet automatically picks the cheapest provider that can handle the job — from a free local model on your Mac to cloud APIs like Kimi, Gemini, or Anthropic. By the end of this guide, you'll have Mullet running inside Claude Code, saving you money on every coding task while keeping responses fast and high-quality.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Install Homebrew](#2-install-homebrew)
- [3. Install Python 310](#3-install-python-310)
- [4. Install Git](#4-install-git)
- [5. Install Claude Code](#5-install-claude-code)
- [6. Get Your API Keys](#6-get-your-api-keys)
  - [6a. Kimi K2 — Recommended Primary Provider](#6a-kimi-k2--recommended-primary-provider)
  - [6b. Anthropic — Optional Orchestrator](#6b-anthropic--optional-orchestrator)
  - [6c. Google Gemini — Optional Fallback](#6c-google-gemini--optional-fallback)
- [7. Install Mullet](#7-install-mullet)
- [8. Set Your API Keys](#8-set-your-api-keys)
- [9. Run mullet doctor](#9-run-mullet-doctor)
- [10. Set Up a Free Local LLM](#10-set-up-a-free-local-llm)
  - [Option A: LM Studio Recommended for Mac newcomers](#option-a-lm-studio-recommended-for-mac-newcomers)
  - [Option B: Ollama Lighter, terminal-based](#option-b-ollama-lighter-terminal-based)
- [11. Wire Mullet into Claude Code](#11-wire-mullet-into-claude-code)
- [12. Test That Everything Works](#12-test-that-everything-works)
- [13. Knowledge Pools Optional — Skip if Overwhelmed](#13-knowledge-pools-optional--skip-if-overwhelmed)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Pro Tip: Planning with Opus](#15-pro-tip-planning-with-opus)
- [16. Next Steps](#16-next-steps)

---

## 1. Before You Start

Here's what you'll need. Don't worry — everything below is explained step by step.

- **macOS 12 Monterey or later.** To check: click the **Apple menu** in the top-left corner of your screen, then choose **About This Mac**. If the version number is 12 or higher, you're good to go.
- **At least 8 GB of RAM** and **20 GB of free disk space.** Local AI models need room to breathe.
- **30–45 minutes** the first time you run through this. (The next time will take two minutes. You're doing great.)
- **A credit card handy** for API keys. Kimi, our recommended primary provider, costs roughly **$1 to get started**.

---

## 2. Install Homebrew

**Homebrew** is a tool that installs other tools for you. Think of it as the Mac App Store, but for the technical programs developers use. We need it to install Python, Git, and a few other things.

First, let's check if it's already on your Mac. Open **Terminal** (press `Cmd + Space`, type `Terminal`, and press `Return`). Then type:

```bash
brew --version
```

If you see something like `Homebrew 4.x.x`, you're all set — skip to [Section 3](#3-install-python-310).

If you see `command not found`, paste this one line into Terminal and press `Return`:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

This will ask for your Mac password (the same one you use to log in). Type it — you won't see any dots on screen for security reasons — and press `Return`.

**What Homebrew does:** It downloads and manages software in the background so you don't have to hunt down installers yourself. This is the only time you'll need to do this.

**Expected output when done:**

```
==> Installation successful!
Run 'brew help' to get started
```

---

## 3. Install Python 3.10+

Mullet is written in **Python**, a popular programming language. macOS comes with an older version, so let's make sure you have a fresh one.

In Terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x` or higher** (for example, `Python 3.12.4`), you're good — skip to [Section 4](#4-install-git).

**If you see `command not found` or a version below 3.10**, run:

```bash
brew install python@3.12
```

Homebrew will download and install Python. This may take a few minutes.

After it finishes, you may see a message about adding Python to your **PATH**. Your PATH is a list of folders your Mac searches when you type a command. If Homebrew tells you to run a command starting with `echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc`, go ahead and paste it into Terminal and press `Return`.

**Verify it worked:**

```bash
python3 --version
```

You should now see `Python 3.12.x` (or similar).

---

## 4. Install Git

**Git** is a tool for downloading and updating code from the internet. We'll use it to fetch Mullet.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set — skip to [Section 5](#5-install-claude-code).

**If macOS pops up a window saying "The git command requires the command line developer tools,"** click **Install**. This is Apple's way of offering you Git — it's perfectly safe, and it only takes a minute.

**If nothing happens or you see `command not found`**, install Git via Homebrew:

```bash
brew install git
```

**Verify it worked:**

```bash
git --version
```

---

## 5. Install Claude Code

**Claude Code** is Anthropic's AI coding assistant that runs in your Terminal. Mullet plugs directly into it to route your questions to the cheapest, fastest AI model for each task.

Claude Code is installed via **npm**, a tool that comes with **Node.js**. Let's check if Node is already installed:

```bash
node --version
```

**If you see `v20.x.x` or higher**, Node is ready.

**If you see `command not found`**, install Node via Homebrew:

```bash
brew install node
```

Now install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
```

The `-g` flag means "install globally" so you can run it from anywhere.

**Verify it worked:**

```bash
claude --version
```

You should see something like `Claude Code 0.x.x`.

**Note:** This is the tool Mullet plugs into. You won't pay Anthropic for every request — Mullet will route most of your questions to cheaper providers instead.

---

## 6. Get Your API Keys

An **API key** is like a password that lets Mullet talk to AI services on your behalf. You'll add these keys to your Mac once, and Mullet will use them automatically.

You only **need** the Kimi key to get started. The others are optional but helpful.

### 6a. Kimi K2 — Recommended Primary Provider

**Kimi** (by Moonshot AI) is Mullet's default workhorse — fast, capable, and affordable.

1. Open your web browser and go to **[platform.moonshot.cn](https://platform.moonshot.cn)**.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add **$5** (about the cost of a coffee). This lasts a surprisingly long time.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-mac` and click **Create**.
6. **Copy the key immediately** — it looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal. That's how you know it's a Kimi key.
7. Paste it somewhere safe (like Apple Notes) for now. We'll add it to your Mac in [Section 8](#8-set-your-api-keys).

**Cost:**

| What you pay | Price |
|---|---|
| Input (what you send) | ~$0.60 per 1 million tokens |
| Output (what you receive) | ~$2.50 per 1 million tokens |

*One million tokens is roughly 750,000 words — about three full novels. For everyday coding questions, you'll spend pennies.*

### 6b. Anthropic — Optional (Orchestrator)

**Anthropic** makes Claude, the AI behind Claude Code. Mullet uses Anthropic's most powerful model (Opus) only for complex planning and orchestration tasks — not for everyday coding.

1. Go to **[console.anthropic.com](https://console.anthropic.com)**.
2. Sign up and add a payment method.
3. Go to **API Keys** → **Create Key**.
4. Copy the key (it also starts with `sk-`).

**Cost:**

| What you pay | Price |
|---|---|
| Input | ~$3.00 per 1 million tokens |
| Output | ~$15.00 per 1 million tokens |

*Don't panic at the price — Mullet sends only a tiny fraction of requests here. It's used sparingly when the orchestrator needs to make smart routing decisions.*

### 6c. Google Gemini — Optional Fallback

**Gemini** (by Google) offers a generous free tier, making it a great safety net.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers most casual usage. If you exceed it, paid rates are very low.

---

## 7. Install Mullet

Now for the main event. These commands download Mullet from the internet and set it up on your Mac.

In Terminal, run these commands one at a time:

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .
```

**What `-e .` means:** The `-e` stands for "editable." It installs Mullet in a special way so that if the code is updated later, you get the changes instantly without reinstalling. The `.` means "install the project in this folder." Think of it like creating a shortcut instead of making a copy.

**Make sure your Mac can find Mullet** — run this now so it works immediately and every time you open Terminal:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Verify it worked:**

```bash
mullet --version
```

You should see something like:

```
mullet 0.x.x
```

---

## 8. Set Your API Keys

Now we'll tell your Mac to remember your API keys so Mullet can use them automatically.

We'll edit a hidden file called **`.zshrc`** using **nano**, a simple text editor that runs inside Terminal. (We use nano instead of other editors because it's friendly for beginners.)

1. Open nano:

```bash
nano ~/.zshrc
```

2. Use the **Down arrow key** to scroll to the very bottom of the file.
3. Add these lines, **replacing the placeholder text with your actual keys**:

```bash
# Mullet API Keys
export KIMI_API_KEY="sk-your-actual-kimi-key-here"
export ANTHROPIC_API_KEY="sk-your-actual-anthropic-key-here"
export GEMINI_API_KEY="your-actual-gemini-key-here"
```

If you only got the Kimi key, that's perfectly fine — just add that one line and skip the others.

4. Save and exit nano:
   - Press `Control + O` (that's the letter O, not zero). Nano will ask "File Name to Write."
   - Press `Return` to confirm.
   - Press `Control + X` to exit.

5. Tell your Terminal to reload the file:

```bash
source ~/.zshrc
```

**Verify your keys are set:**

```bash
echo $KIMI_API_KEY
```

You should see your key printed on the screen (something starting with `sk-`). If you see a blank line, double-check that you saved the file correctly.

**Do the same for the other keys if you added them:**

```bash
echo $ANTHROPIC_API_KEY
echo $GEMINI_API_KEY
```

---

## 9. Run mullet doctor

Mullet comes with a built-in health check that tells you whether everything is set up correctly.

Run:

```bash
mullet doctor
```

You'll see output like this:

```
✓ Python 3.12 found
✓ Git installed
✓ Kimi API key valid
! Anthropic API key not set (optional)
· Local LLM not detected (optional)
✗ Node.js not found
```

**Here's what each symbol means:**

| Symbol | Color | Meaning |
|---|---|---|
| ✓ | Green | All good! This check passed. |
| ! | Yellow | Warning, not an error. This is optional or needs attention later. |
| · | Cyan | Informational. Something isn't set up yet, but it's okay to skip for now. |
| ✗ | Red | Problem that needs fixing before Mullet works properly. |

At the bottom, you'll see a line like:

```
5/7 checks passed
```

**Don't panic if some things are yellow.** Yellow items are optional. Red items are the ones to fix. If you see red, jump to the [Troubleshooting](#14-troubleshooting) section for the fix.

---

## 10. Set Up a Free Local LLM

A **local LLM** is an AI model that runs entirely on your own Mac — no internet required, no API costs, and your code stays private. Mullet can route simple questions to your local model for free, reserving your paid API keys for harder tasks.

You have two options. **Pick one.**

### Option A: LM Studio (Recommended for Mac newcomers)

**LM Studio** is a normal Mac app with a friendly interface. This is the easiest way to get a local model running.

1. **Download LM Studio:**
   - Go to **[lmstudio.ai](https://lmstudio.ai)** in your web browser.
   - Click **Download for Mac**.
   - Open the downloaded `.dmg` file.

2. **Install LM Studio:**
   - Drag the **LM Studio** icon into your **Applications** folder (just like any other Mac app).
   - Open **LM Studio** from your Applications folder.

3. **Download a model:**
   - In LM Studio, click the **search icon** on the left sidebar (it looks like a magnifying glass).
   - In the search bar at the top, type: **`Qwen2.5-Coder-7B-Instruct`**
   - Click the model name in the results, then click **Download**.
   - This downloads about **4 GB** of data. Grab a coffee — it may take 10–20 minutes depending on your internet.
   - **What this model is:** A 7-billion-parameter AI trained specifically for coding tasks. It's small enough to run smoothly on most Macs but smart enough to handle simple coding questions.

4. **Load the model:**
   - Once downloaded, click the **chat icon** on the left sidebar.
   - At the top of the chat panel, click the model dropdown and select **Qwen2.5-Coder-7B-Instruct**.
   - Click the **Load** button.

5. **Start the local server:**
   - Look for the **server toggle switch** in the top bar of LM Studio (it says "Server" with an on/off switch).
   - **Click the toggle to turn it ON.**
   - The default port is **1234**. You don't need to change this.

6. **Verify:**
   - Go back to Terminal and run:

```bash
mullet doctor
```

   - You should now see a green **✓ Local LLM** line.

**Screenshot callouts:**
- **Search bar:** Top of the search panel (magnifying glass icon).
- **Server toggle:** Top center of the main window, labeled "Server."

### Option B: Ollama (Lighter, terminal-based)

**Ollama** is a lightweight tool that runs models from the command line. It's great if you prefer keeping everything in Terminal.

Install it via Homebrew:

```bash
brew install ollama
```

Download the same coding model:

```bash
ollama pull qwen2.5-coder:7b
```

This downloads about 4 GB. Once it finishes, start the server:

```bash
ollama serve
```

This runs in the foreground. **Leave this Terminal window open.** If you close it, the local model stops.

*(To run it in the background later, you can set it up as a Mac service, but that's outside the scope of this guide.)*

**Verify:**

Open a *new* Terminal window (`Cmd + N`) and run:

```bash
mullet doctor
```

You should see a green **✓ Local LLM** line. Ollama runs on port **11434**, and Mullet detects it automatically.

---

## 11. Wire Mullet into Claude Code

This is the magic step. One command installs Mullet's hooks, MCP server, delegation rules, skills, agents, and status bar into Claude Code.

Run:

```bash
mullet setup-claude
```

**What this does:**
- Installs the **MCP server** so Claude Code can call Mullet's tools
- Adds **delegation rules** so Claude automatically routes tasks through Mullet
- Copies **skills** and **agents** into Claude Code's configuration
- Sets up the **status bar HUD** so you can see which provider is being used in real time

**Important:** After this command finishes, you must **fully quit and reopen Claude Code**.

- **Don't just close the window.** Press `Cmd + Q` to quit completely, then reopen Claude Code.
- If you're running Claude Code inside Terminal, press `Control + C` to exit, then type `claude` again to restart.

---

## 12. Test That Everything Works

Let's run three quick tests to confirm Mullet is alive and routing correctly.

### Test 1 — CLI works

Ask Mullet a simple question directly from Terminal:

```bash
mullet "what Python version am I running?"
```

**Expected output:**

You should see an answer within a few seconds, routed to either your **local LLM** (if it's running) or **Kimi** (if not). At the top of the output, you may see a line telling you which provider was chosen.

### Test 2 — See routing before spending money

Mullet can show you what it *would* do without actually making an expensive API call:

```bash
mullet explain "refactor the authentication module"
```

**Expected output:**

```
Complexity: medium
Provider: Kimi K2
Estimated cost: $0.0042
```

This tells you:
- How hard Mullet thinks your request is
- Which AI it would send it to
- About how much it would cost

No actual LLM call is made during `explain`, so this test is completely free.

### Test 3 — MCP works inside Claude Code

1. Open Claude Code in any project folder:

```bash
claude
```

2. At the Claude prompt, type exactly:

```
Use mullet_run to summarise what this project does.
```

**Expected output:**

Claude should recognize the `mullet_run` tool and display a **tool call** in the chat. You may see a box or panel showing that Claude is invoking a Mullet MCP tool. If that happens, congratulations — Mullet is fully wired into Claude Code.

If Claude says it doesn't know that tool, see [Troubleshooting](#14-troubleshooting).

---

## 13. Knowledge Pools (Optional — Skip if Overwhelmed)

**Knowledge Pools** let Mullet index your codebase so Claude can search it intelligently instead of pasting entire files into the conversation.

If you're feeling overwhelmed, **skip this for now.** You can always come back to it later.

**Quick flow:**

1. In your project folder, run:

```bash
mullet init
```

2. Index your code:

```bash
mullet reindex --pool myproject
```

3. Inside Claude Code, ask:

```
Use trawl_search to find where the user authentication is handled.
```

Claude will use Mullet's semantic search to find the right files without you having to tell it where to look.

For full details, see the **[SKILLS.md](SKILLS.md)** document.

---

## 14. Troubleshooting

Something not working? You're not alone. Here's a quick lookup table.

| Problem | Fix |
|---|---|
| `mullet: command not found` after install | Add `~/.local/bin` to your PATH. Open `nano ~/.zshrc`, scroll to the bottom, and add:<br>`export PATH="$HOME/.local/bin:$PATH"`<br>Save (`Control + O`, `Return`, `Control + X`), then run `source ~/.zshrc`. |
| `pip: command not found` | Use `pip3` instead of `pip`, or reinstall Python via `brew install python@3.12`. |
| `mullet doctor` shows no providers | Check that your keys are exported. Run `echo $KIMI_API_KEY` — if it prints nothing, revisit [Section 8](#8-set-your-api-keys). |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Cmd + Q`) and reopen it. Simply closing the window is not enough. |
| LM Studio detected but context warning | Load a **7B+ model** like Qwen2.5-Coder-7B. Also check the **Context Length** slider in LM Studio's model settings — set it to at least 4096. |
| Kimi key rejected | Make sure your key starts with `sk-`. Check that you have credits at **[platform.moonshot.cn](https://platform.moonshot.cn)**. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong (often a missing Python package). |
| `python3 -m pip install` fails with permissions error | Add the `--user` flag:<br>`pip install --user -e .` |

Still stuck? Run the full diagnostic and share the output:

```bash
mullet doctor --full
```

---

## 15. Pro Tip: Planning with Opus

Before starting any big coding task in Claude Code, try this command:

```
/model opusplan
```

Type it in the Claude Code chat window. This switches Claude to **Opus** (Anthropic's most powerful model) just for the planning phase, so you get a thorough, well-thought-out plan before any code is written. Once the plan is approved and implementation starts, Mullet automatically routes the actual code work to cheaper models (Kimi or local LLM) — so you get Opus-quality thinking without paying Opus prices for every line of code.

**When to use it:**
- Starting a new feature from scratch
- Tackling a complex bug you don't fully understand yet
- Refactoring a large section of code
- Any task where getting the approach right matters more than speed

It's one of the most powerful combos in the Mullet toolkit.

---

## 16. Next Steps

You're all set! Mullet is now saving you money on every AI-assisted coding task.

**Where to go from here:**

- **Quick reference for daily commands:** See **[QUICKSTART.md](QUICKSTART.md)**
- **How Mullet routes requests and chooses providers:** See **[ARCHITECTURE.md](ARCHITECTURE.md)**
- **Full installation reference (including Linux):** See **[INSTALL.md](INSTALL.md)**
- **Something broken?** Run `mullet doctor --full` and paste the output into a GitHub issue.

**Welcome to the Mullet family.** Happy coding — and happy saving.
