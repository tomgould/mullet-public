# Mullet on Mac: Complete Setup Guide

> **Mullet runs large coding jobs unattended and checks every step itself.** It also gives Claude Code a knowledge-search server (Trawl) and optional tools that hand work to cheaper models (Kimi, OpenRouter, Gemini, or a model on your own Mac).
>
> **What works on a Mac today:** the MCP server, Trawl search, the delegation tools and local models. **`mullet job` does not start on macOS yet.** Jobs run inside a bubblewrap sandbox, which is Linux-only, and a macOS sandbox is coming but not built. Use a Linux machine to run jobs. This guide sets up everything else.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Install Homebrew](#2-install-homebrew)
- [3. Install Python 3.10 to 3.12](#3-install-python-310-to-312)
- [4. Install Git](#4-install-git)
- [5. Install Claude Code](#5-install-claude-code)
- [6. Get Your API Keys (Optional)](#6-get-your-api-keys-optional)
  - [6a. Kimi K2 (Recommended)](#6a-kimi-k2-recommended)
  - [6b. Anthropic (Optional)](#6b-anthropic-optional)
  - [6c. Google Gemini (Optional)](#6c-google-gemini-optional)
- [7. Install Mullet](#7-install-mullet)
- [8. Set Your API Keys](#8-set-your-api-keys)
- [9. Run mullet doctor](#9-run-mullet-doctor)
- [10. Set Up a Free Local LLM](#10-set-up-a-free-local-llm)
  - [Option A: LM Studio Recommended for Mac newcomers](#option-a-lm-studio-recommended-for-mac-newcomers)
  - [Option B: Ollama Lighter, terminal-based](#option-b-ollama-lighter-terminal-based)
- [11. Wire Mullet into Claude Code](#11-wire-mullet-into-claude-code)
- [12. Test That Everything Works](#12-test-that-everything-works)
- [13. Knowledge Pools (Optional, Skip if Overwhelmed)](#13-knowledge-pools-optional-skip-if-overwhelmed)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Pro Tip: Planning with Opus](#15-pro-tip-planning-with-opus)
- [16. Next Steps](#16-next-steps)

---

## 1. Before You Start

Here's what you'll need. Don't worry: everything below is explained step by step.

- **macOS 12 Monterey or later.** To check: click the **Apple menu** in the top-left corner of your screen, then choose **About This Mac**. If the version number is 12 or higher, you're good to go.
- **At least 8 GB of RAM** and **20 GB of free disk space** if you want a local model. Local AI models need room to breathe. Mullet itself is small.
- **30 to 45 minutes** the first time you run through this. (The next time will take two minutes. You're doing great.)
- **A Claude Code login** (a Claude subscription). Mullet works alongside it.
- **A credit card handy**, only if you want cloud API keys. They are optional. Kimi, the recommended one, costs roughly **$1 to get started**. Trawl search and local models need no key at all.

---

## 2. Install Homebrew

**Homebrew** is a tool that installs other tools for you. Think of it as the Mac App Store, but for the technical programs developers use. We need it to install Python, Git, and a few other things.

First, let's check if it's already on your Mac. Open **Terminal** (press `Cmd + Space`, type `Terminal`, and press `Return`). Then type:

```bash
brew --version
```

If you see something like `Homebrew 4.x.x`, you're all set. Skip to [Section 3](#3-install-python-310-to-312).

If you see `command not found`, paste this one line into Terminal and press `Return`:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

This will ask for your Mac password (the same one you use to log in). Type it (you won't see any dots on screen, for security reasons) and press `Return`.

**What Homebrew does:** It downloads and manages software in the background so you don't have to hunt down installers yourself. This is the only time you'll need to do this.

**Expected output when done:**

```
==> Installation successful!
Run 'brew help' to get started
```

---

## 3. Install Python 3.10 to 3.12

Mullet is written in **Python**, a popular programming language. macOS comes with an older version, so let's make sure you have a fresh one.

In Terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x`, `3.11.x` or `3.12.x`** (for example, `Python 3.12.4`), you're good. Skip to [Section 4](#4-install-git).

**If you see `command not found` or a version outside 3.10 to 3.12**, run:

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

**Git** is a tool for downloading and updating code from the internet. The Mullet installer uses it to fetch Mullet from GitHub.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set. Skip to [Section 5](#5-install-claude-code).

**If macOS pops up a window saying "The git command requires the command line developer tools,"** click **Install**. This is Apple's way of offering you Git. It's perfectly safe, and it only takes a minute.

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

**Claude Code** is Anthropic's AI coding assistant that runs in your Terminal. Mullet plugs directly into it: it adds search over your code and docs, and optional tools that hand work to cheaper models.

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

You should see a version number followed by `(Claude Code)`.

Then run `claude` once and log in with your Claude account when it asks.

**Note:** Mullet never reads or reuses your Claude login. Claude Code keeps it, and Mullet's jobs run your own logged-in `claude`.

---

## 6. Get Your API Keys (Optional)

An **API key** is like a password that lets Mullet talk to AI services on your behalf. Keys are **optional**. You only need them for Mullet's delegation tools, which hand work to cheaper models. Trawl search, `mullet doctor` and a local model work without any key.

If you want to try delegation, start with the Kimi key. The others are optional extras. You'll add the keys to one file on your Mac in [Section 8](#8-set-your-api-keys).

### 6a. Kimi K2 (Recommended)

**Kimi** (by Moonshot AI) is the usual first choice for delegated work: fast, capable, and affordable.

1. Open your web browser and go to **[platform.moonshot.ai](https://platform.moonshot.ai)**.
   - If you are in mainland China, use [platform.moonshot.cn](https://platform.moonshot.cn) and add `KIMI_BASE_URL=https://api.moonshot.cn/v1` to your `.env` file in [Section 8](#8-set-your-api-keys). Mullet uses the international address by default.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add **$5** (about the cost of a coffee). This lasts a surprisingly long time.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-mac` and click **Create**.
6. **Copy the key immediately.** It looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal.
7. Paste it somewhere safe (like Apple Notes) for now. We'll add it to your Mac in [Section 8](#8-set-your-api-keys).

**Cost:** Kimi charges per token (a token is roughly three-quarters of a word). Prices change, so check the pricing page on the Moonshot site. For everyday coding questions, you'll spend pennies.

### 6b. Anthropic (Optional)

**Anthropic** makes Claude, the AI behind Claude Code. You do **not** need an Anthropic API key to use Claude Code or Mullet. Claude Code uses your subscription login.

An Anthropic key is billed per token, separately from your subscription. If it is set when Mullet runs `claude -p`, that call bills the API instead of your subscription. Most people should skip this key. Mullet's jobs leave it out unless you plan one with `--billing api`, and `mullet doctor` warns you when it is set.

If you do want one:

1. Go to **[console.anthropic.com](https://console.anthropic.com)**.
2. Sign up and add a payment method.
3. Go to **API Keys**, then **Create Key**.
4. Copy the key (it also starts with `sk-`).

### 6c. Google Gemini (Optional)

**Gemini** (by Google) offers a free tier, which makes it a handy extra provider.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers most casual usage. Check Google's pricing page for the paid rates.

Gemini also needs a small extra Python package. [Section 7](#7-install-mullet) shows how to add it.

---

## 7. Install Mullet

Now for the main event. One command downloads Mullet from GitHub and installs it as its own tool, so it can't clash with other Python projects on your Mac.

The installer needs **git** and **Python 3.10 to 3.12** (you have both from the earlier sections), plus **uv** or **pipx**. If you have neither, install uv:

```bash
brew install uv
```

Then run the installer:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

To update Mullet later, run the same command again.

**Optional extras.** To add Gemini support, set `MULLET_EXTRAS` for the run:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | MULLET_EXTRAS=gemini bash
```

The extras are `gemini`, `bedrock` and `postgres`. **Avoid `all` on a Mac**: it includes the NVIDIA-only `pynvml` package, which won't install here. Don't use the `gpu` extra either.

**Never run `pip install mullet`.** The package of that name on PyPI is someone else's project, not this one.

**If `mullet` is not found** after the install, your Mac doesn't yet look in the folder where uv puts tools. Run:

```bash
uv tool update-shell
```

(If you used pipx, run `pipx ensurepath` instead.) Then open a **new Terminal window**.

**Verify it worked:**

```bash
mullet --version
```

You should see `mullet` followed by the version number.

> **Developer note:** If you want to change Mullet's own code, clone the repository instead, create a virtual environment, and run `pip install -e ".[dev]"`. Ordinary users should use the installer above.

---

## 8. Set Your API Keys

Skip this section if you didn't get any keys. Mullet works without them for Trawl search and local models.

Mullet reads its keys from one file: **`~/.config/mullet/.env`**. Put them there, not in `~/.zshrc`. (If you set `MULLET_CONFIG_DIR`, the file is `$MULLET_CONFIG_DIR/.env` instead.)

1. Create the folder and the file, and make the file private to you:

```bash
mkdir -p ~/.config/mullet
touch ~/.config/mullet/.env
chmod 600 ~/.config/mullet/.env
```

2. Open it in **nano**, a simple text editor that runs inside Terminal:

```bash
nano ~/.config/mullet/.env
```

3. Add a line for each key you have, **replacing the placeholder text with your actual key**. Use plain `NAME=value` lines, with no `export` and no quotes:

```
KIMI_API_KEY=sk-your-actual-kimi-key-here
OPENROUTER_API_KEY=your-actual-openrouter-key-here
GOOGLE_API_KEY=your-actual-gemini-key-here
```

If you only got the Kimi key, that's perfectly fine. Add just that line. `ANTHROPIC_API_KEY` is also read from this file, but see [Section 6b](#6b-anthropic-optional) before adding it.

4. Save and exit nano:
   - Press `Control + O` (that's the letter O, not zero). Nano will ask "File Name to Write."
   - Press `Return` to confirm.
   - Press `Control + X` to exit.

You don't need to restart Terminal. Mullet reads the file each time it runs.

**How the file is read:**
- A variable already set in your Terminal wins over the file.
- A `.env` file inside one of your own projects is never read.

**Verify a key is set without printing it:**

```bash
grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env
```

`1` means the line is there. `0` means it isn't. Avoid `echo $KIMI_API_KEY`: it prints your key on screen, and anyone looking over your shoulder (or any screen share) would see it.

Repeat with `OPENROUTER_API_KEY` or `GOOGLE_API_KEY` for the others.

---

## 9. Run mullet doctor

Mullet comes with a built-in health check that tells you whether everything is set up correctly. It only reads your setup and works offline, so it is safe to run at any time.

Run:

```bash
mullet doctor
```

You'll see output like this:

```
  ✓ Python               3.12.4
  ✓ Claude Code CLI      2.1.302 (Claude Code)
  ✓ Git                  2.39.5
  · Job sandbox          Unavailable: no sandbox backend for darwin yet
  · Local LLM            Not detected (optional)
  ✓ Kimi API             KIMI_API_KEY set
  · Google Gemini        GOOGLE_API_KEY not set
  · Claude Code setup    Not set up: run mullet setup-claude
```

The exact lines depend on your setup. At the bottom you'll see a count such as `21/24 checks passed`.

**Here's what each symbol means:**

| Symbol | Color | Meaning |
|---|---|---|
| ✓ | Green | All good! This check passed. |
| ! | Yellow | Warning, not an error. This is optional or needs attention later. |
| · | Cyan | Informational. Something isn't set up yet, but it's okay to skip for now. |
| ✗ | Red | Problem that needs fixing before Mullet works properly. |

**The job sandbox line:** On a Mac it will say the sandbox is unavailable. That is expected. It is why `mullet job` won't start here.

**A key warning:** If you see a yellow `ANTHROPIC_API_KEY` line, a key is set in your environment and `claude -p` will bill the API instead of your subscription. Remove it if you didn't mean that.

**Don't panic if some things are yellow.** Yellow items are optional. Red items are the ones to fix. If you see red, jump to the [Troubleshooting](#14-troubleshooting) section.

For a deeper check, run `mullet doctor --full`. It also probes your providers and asks GitHub whether a newer Mullet exists.

---

## 10. Set Up a Free Local LLM

A **local LLM** is an AI model that runs entirely on your own Mac: no internet required, no API costs, and your code stays private. Mullet can hand simple questions to your local model for free. This section is optional.

**Pick a model that fits your RAM.** Open **About This Mac** to check.

| Your Mac's RAM | Model to use |
|---|---|
| 10 GB or less | A 2B to 3B model. Don't try 7B or larger: it will be slow and may freeze your Mac. |
| 16 GB or more | `Qwen2.5-Coder-7B-Instruct` (used in the steps below). |

The steps below use the 7B model. On a smaller Mac, search for a 2B to 3B coding model instead and follow the same steps.

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
   - This downloads about **4 GB** of data. Grab a coffee. It may take 10 to 20 minutes depending on your internet.
   - **What this model is:** A 7-billion-parameter AI trained specifically for coding tasks. It's small enough to run on a Mac with 16 GB of RAM or more, and smart enough to handle simple coding questions.

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

Download the same coding model (on a Mac with 10 GB of RAM or less, pull a 2B to 3B model instead):

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

One command registers Mullet's MCP server and skills with Claude Code. Run it with `--dry-run` first to see exactly what it will change:

```bash
mullet setup-claude --dry-run
```

If you're happy with the list, run it for real:

```bash
mullet setup-claude
```

**What this does:**
- Registers the **MCP server** for every project (`claude mcp add --scope user mullet -- <mullet> mcp`), so Claude Code can call Mullet's tools
- Installs the `mullet-job` and `pool-improve` **skills** into `~/.claude/skills/`
- Adds **hooks** only if you ask: `mullet setup-claude --hooks list` describes them, and `--hooks name1,name2` adds the ones you name
- Sets the status bar only if you ask: `--statusline`
- Never edits your `CLAUDE.md`. `mullet setup-claude --uninstall` removes exactly what it added

**Important:** After this command finishes, you must **fully quit and reopen Claude Code**.

- **Don't just close the window.** Press `Cmd + Q` to quit completely, then reopen Claude Code.
- If you're running Claude Code inside Terminal, press `Control + C` to exit, then type `claude` again to restart.

---

## 12. Test That Everything Works

Let's run three quick tests to confirm Mullet is alive and working.

### Test 1: CLI works

Ask Mullet a simple question directly from Terminal. This needs a provider, so set up a key ([Section 8](#8-set-your-api-keys)) or a local model ([Section 10](#10-set-up-a-free-local-llm)) first:

```bash
mullet "what Python version am I running?"
```

**Expected output:**

You should see an answer within a few seconds, routed to either your **local LLM** (if it's running) or a cloud provider you have a key for. Add `--verbose` to see which provider was chosen.

### Test 2: See routing before spending money

Mullet can show you what it *would* do without actually making an expensive API call:

```bash
mullet explain "refactor the authentication module"
```

**Expected output:**

```
Routing Analysis: "refactor the authentication module"

  Simple message: No
  Complexity score: 80.0/100
  Routing tier: ADVANCED
  ...
  Provider: Premium cloud (Sonnet/Opus)
  Estimated cost: $0.0210
```

The exact numbers vary. This tells you:
- How hard Mullet thinks your request is
- Which kind of model it would send it to
- About how much it would cost

No actual LLM call is made during `explain`, so this test is completely free.

### Test 3: MCP works inside Claude Code

1. Open Claude Code in any project folder:

```bash
claude
```

2. At the Claude prompt, type exactly:

```
Use trawl_list_pools to show my knowledge pools.
```

**Expected output:**

Claude should recognise the `trawl_list_pools` tool and display a **tool call** in the chat. The list may be empty if you haven't made a pool yet. If Claude calls the tool, congratulations: Mullet is fully wired into Claude Code.

To try a delegation tool, you need a key or local model. Ask: `Use mullet_run to summarise what this project does.`

If Claude says it doesn't know the tool, see [Troubleshooting](#14-troubleshooting).

---

## 13. Knowledge Pools (Optional, Skip if Overwhelmed)

**Knowledge Pools** let Mullet index your codebase so Claude can search it intelligently instead of pasting entire files into the conversation. This is Trawl, Mullet's search engine. It works on a Mac.

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

Pools are stored in the folder that `mullet paths pools` prints. Set `TRAWL_POOLS_DIR` to keep them somewhere else.

For full details, see the **[SKILLS.md](SKILLS.md)** document.

---

## 14. Troubleshooting

Something not working? You're not alone. Here's a quick lookup table.

| Problem | Fix |
|---|---|
| `mullet: command not found` after install | Run `uv tool update-shell` (or `pipx ensurepath` if you used pipx), then open a **new Terminal window**. |
| The installer says it needs uv or pipx | Run `brew install uv`, then run the installer again. |
| The installer says git is required | Follow [Section 4](#4-install-git), then run the installer again. |
| `mullet job` says it can't start on macOS | Expected. Jobs need the Linux sandbox. Run jobs on a Linux machine. |
| `pip install` fails with a Gemini or `pynvml` error | You asked for the `all` or `gpu` extra. Reinstall with `MULLET_EXTRAS=gemini` (see [Section 7](#7-install-mullet)). |
| `mullet doctor` shows no providers | Check that your keys are in `~/.config/mullet/.env`. Run `grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env`. If it prints `0`, revisit [Section 8](#8-set-your-api-keys). Keys in `~/.zshrc` still work as environment variables, but the file is the supported place. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Cmd + Q`) and reopen it. Simply closing the window is not enough. |
| LM Studio detected but context warning | Check the **Context Length** slider in LM Studio's model settings and set it to at least 4096. On a Mac with more than 10 GB of RAM, a 7B model like Qwen2.5-Coder-7B works well. On 10 GB or less, stay with a 2B to 3B model. |
| Kimi key rejected | Check that you have credits at **[platform.moonshot.ai](https://platform.moonshot.ai)**. If your key came from the `.cn` site, set `KIMI_BASE_URL=https://api.moonshot.cn/v1` in `~/.config/mullet/.env`. |
| Claude Code bills the API, not my subscription | `ANTHROPIC_API_KEY` is set somewhere. `mullet doctor` flags it. Remove it from your shell profile or `~/.config/mullet/.env`. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong. |
| Older `~/.mullet` folder still in use | It keeps working. To move it to the new locations, quit Claude Code, run `mullet migrate-state --dry-run` to preview, then `mullet migrate-state`. |

Still stuck? Run the full diagnostic and share the output:

```bash
mullet doctor --full
```

Mullet keeps its files in these places. `mullet paths` prints them:

| What | Where |
|---|---|
| Config and keys | `~/.config/mullet` |
| State | `~/.local/state/mullet` |
| Data (pools, skills, jobs) | `~/.local/share/mullet` |
| Caches | `~/.cache/mullet` |

---

## 15. Pro Tip: Planning with Opus

Before starting any big coding task in Claude Code, try this command:

```
/model opusplan
```

Type it in the Claude Code chat window. This makes Claude Code use **Opus** (Anthropic's most powerful model) for the planning phase and a cheaper model while it writes the code. You get a thorough plan before any code is written, without Opus prices for every line.

This is a Claude Code feature, not a Mullet one. Mullet adds to it: when you ask, Claude can use Mullet's delegation tools to hand a task to Kimi, OpenRouter or a local model.

**When to use it:**
- Starting a new feature from scratch
- Tackling a complex bug you don't fully understand yet
- Refactoring a large section of code
- Any task where getting the approach right matters more than speed

---

## 16. Next Steps

You're all set. Mullet now gives Claude Code search over your code and optional cheaper-model tools.

**Where to go from here:**

- **Quick reference for daily commands:** See **[QUICKSTART.md](QUICKSTART.md)**
- **How Mullet routes requests and chooses providers:** See **[ARCHITECTURE.md](ARCHITECTURE.md)**
- **Full installation reference (including Linux):** See **[INSTALL.md](INSTALL.md)**
- **Run `mullet job` on a Linux machine:** `mullet job plan "<goal>"` makes a task tree to approve. Then `approve`, `run`, `status`, `tail -f`, `report` and `merge <id>` take it through to a branch and an evidence report. See **[ONBOARDING_LINUX.md](ONBOARDING_LINUX.md)**.
- **Coming, not built yet:** job sandboxing on macOS, a ladder of Claude, OpenRouter, Kimi, free and local models for job steps, and `mullet job bench`.
- **Something broken?** Run `mullet doctor --full` and paste the output into a GitHub issue.

Happy coding.
