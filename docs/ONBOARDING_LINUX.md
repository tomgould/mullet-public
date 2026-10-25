# Mullet on Linux: Complete Setup Guide

> **Mullet runs large coding jobs for you and checks every step itself.** You give it a goal, approve the task tree it plans, and it works through the steps unattended. Each step runs with your own logged-in Claude Code in a sandboxed private clone of your repo, and passes only when Mullet's own checks (the gates) pass. You get back a branch and an evidence report, and your checkout is untouched until you merge. Jobs need no API keys. The second, optional part of Mullet routes questions to cheaper models (Kimi, OpenRouter, Gemini or a free local model) from inside Claude Code. By the end of this guide you will have Mullet installed, checked and wired into Claude Code.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Check Your Shell](#2-check-your-shell)
- [3. Install Python 3.10 to 3.12](#3-install-python-310-to-312)
- [4. Install Git](#4-install-git)
- [5. Install Node.js via nvm](#5-install-nodejs-via-nvm)
- [6. Install Claude Code](#6-install-claude-code)
- [7. Get Provider API Keys (Optional)](#7-get-provider-api-keys-optional)
  - [7a. Kimi K2: Recommended Provider for Delegation](#7a-kimi-k2-recommended-provider-for-delegation)
  - [7b. Anthropic: Optional, Billed Per Token](#7b-anthropic-optional-billed-per-token)
  - [7c. Google Gemini: Optional Fallback](#7c-google-gemini-optional-fallback)
- [8. Install Mullet](#8-install-mullet)
- [9. Set Your API Keys (Optional)](#9-set-your-api-keys-optional)
- [10. Run mullet doctor](#10-run-mullet-doctor)
- [11. Set Up a Local LLM (Optional)](#11-set-up-a-local-llm-optional)
  - [Option A: LM Studio (Recommended for GUI lovers)](#option-a-lm-studio-recommended-for-gui-lovers)
  - [Option B: Ollama (Lighter, terminal-based)](#option-b-ollama-lighter-terminal-based)
- [12. Wire Mullet into Claude Code](#12-wire-mullet-into-claude-code)
- [13. Test That Everything Works](#13-test-that-everything-works)
- [14. Knowledge Pools (Optional)](#14-knowledge-pools-optional)
- [15. Pro Tip: Planning with Opus](#15-pro-tip-planning-with-opus)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Next Steps](#17-next-steps)

---

## 1. Before You Start

Here's what you'll need. Everything below is explained step by step.

- **Ubuntu 22.04 or later, or an equivalent Linux distribution.** (Ubuntu 20.04 ships Python 3.8, which is too old.) These instructions focus on **Ubuntu** and **Debian**, which are the most common starting points for newcomers. If you use Fedora, RHEL, or Arch, we've included quick callouts for you too.
- **A Claude Code account.** Mullet jobs run with your own logged-in Claude Code, so a Claude subscription (or Anthropic API access) is the one thing you cannot skip. Jobs need no other API keys.
- **At least 8 GB of RAM** and **20 GB of free disk space.** Local AI models are optional, but they need room. With 10 GB of RAM or less, use a small 2B to 3B model (see [Section 11](#11-set-up-a-local-llm-optional)) or skip local models.
- **A GPU is optional.** If you have an NVIDIA or AMD graphics card with 8 GB or more of video memory (VRAM), you can run larger local models for free and get faster responses. If not, Mullet can use cloud providers instead.
- **30 to 45 minutes** the first time you run through this. The next time will take two minutes.
- **A credit card, only if you want cloud providers.** Kimi, the recommended provider for delegation, costs roughly **$5 to get started**. Skip this if you only want jobs.

---

## 2. Check Your Shell

Your **shell** is the program that interprets the commands you type into the terminal. On Ubuntu, the default is **bash**, but many Linux users prefer **zsh**. The shell you use determines which hidden file we edit for PATH settings.

In your terminal, run:

```bash
echo $SHELL
```

**If you see `/bin/bash` or `/usr/bin/bash`**, you use **bash**. Where this guide touches a shell config file, it means `~/.bashrc`.

**If you see `/bin/zsh` or `/usr/bin/zsh`**, you use **zsh**. We'll edit `~/.zshrc` instead.

**If you see something else** (like `/bin/fish`), we recommend switching to bash or zsh for this guide, or adapting the config file path to your shell.

> **Note for zsh users:** Whenever this guide says `~/.bashrc`, use `~/.zshrc` instead. Whenever it says `source ~/.bashrc`, run `source ~/.zshrc` instead.

---

## 3. Install Python 3.10 to 3.12

Mullet is written in **Python**, a popular programming language. Most Linux distributions come with Python already installed, but let's make sure it's new enough. The installer needs Python 3.10 to 3.12.

In your terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x`, `3.11.x` or `3.12.x`** (for example, `Python 3.12.4`), you're good. Skip to [Section 4](#4-install-git).

**If you see `command not found` or a version outside 3.10 to 3.12**, install Python via your package manager:

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y python3 python3-pip
```

> **Fedora / RHEL:** `sudo dnf install -y python3 python3-pip`
>
> **Arch:** `sudo pacman -S python python-pip`

You may be asked for your password. Type it and press `Return`. You won't see any dots on screen, which is normal.

**Verify it worked:**

```bash
python3 --version
```

You should now see `Python 3.10.x`, `3.11.x` or `3.12.x`.

---

## 4. Install Git

**Git** is a tool for downloading and updating code from the internet. The Mullet installer uses it to fetch Mullet, and Mullet jobs use it to make their private clone of your repo.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set. Skip to [Section 5](#5-install-nodejs-via-nvm).

**If you see `command not found`**, install Git via your package manager:

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y git
```

> **Fedora / RHEL:** `sudo dnf install -y git`
>
> **Arch:** `sudo pacman -S git`

**Verify it worked:**

```bash
git --version
```

---

## 5. Install Node.js via nvm

**Claude Code** is installed with **npm**, a tool that comes with **Node.js**. Node.js lets you run JavaScript programs, and npm installs them. (Mullet itself is pure Python and does not need Node.js. Claude Code does.)

**Important:** The version of Node.js that comes from `apt` is usually very old. We'll use **nvm** (Node Version Manager) instead, which gets you the latest version.

First, check if Node is already installed:

```bash
node --version
```

**If you see `v20.x.x` or higher**, Node is ready.

**If you see `command not found` or an older version**, install nvm:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

This downloads and installs nvm. **Close the terminal window completely and reopen it** so nvm activates. (Type `exit` or close the window, then open a new terminal.)

Now install the latest Long Term Support (LTS) version of Node.js:

```bash
nvm install --lts
```

**Verify it worked:**

```bash
node --version
```

You should see `v20.x.x` or higher.

---

## 6. Install Claude Code

**Claude Code** is Anthropic's AI coding assistant that runs in your terminal. Mullet jobs run through it, and Mullet's tools plug into it.

Install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
```

The `-g` flag means "install globally" so you can run it from anywhere.

**Verify it worked:**

```bash
claude --version
```

You should see a version number followed by `(Claude Code)`.

Now log in. Run `claude`, follow the prompts to sign in, then quit with `Control + C` twice. Mullet jobs use this login, so they bill your subscription and need no API key.

---

## 7. Get Provider API Keys (Optional)

An **API key** is like a password that lets Mullet talk to AI services on your behalf.

**You can skip this whole section if you only want to run jobs.** Jobs use your logged-in Claude Code and need no keys. Keys are for Mullet's delegation tools, which send questions from Claude Code to cheaper models. If you want those, Kimi is the one to start with. The others are optional.

### 7a. Kimi K2: Recommended Provider for Delegation

**Kimi** (by Moonshot AI) is the usual first choice for delegation: fast, capable and affordable.

1. Open your web browser and go to **[platform.moonshot.cn](https://platform.moonshot.cn)**.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add **$5** (about the cost of a coffee). This lasts a surprisingly long time.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-linux` and click **Create**.
6. **Copy the key immediately.** It looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal.
7. Paste it somewhere safe, such as a password manager, for now. We'll add it to your Linux machine in [Section 9](#9-set-your-api-keys-optional).

**Cost:**

| What you pay | Price |
|---|---|
| Input (what you send) | ~$0.60 per 1 million tokens |
| Output (what you receive) | ~$2.50 per 1 million tokens |

*One million tokens is roughly 750,000 words, about three full novels. For everyday coding questions, you'll spend pennies.*

### 7b. Anthropic: Optional, Billed Per Token

**Anthropic** makes Claude, the AI behind Claude Code. Most people do **not** need an Anthropic API key. Claude Code uses your subscription login, and so do Mullet jobs.

**Warning:** if `ANTHROPIC_API_KEY` is set, `claude -p` (which jobs use) bills the API per token instead of your subscription. Leave it unset for jobs. `mullet doctor` warns you if it is set. Only get a key if you want Mullet to call Claude models through the API.

1. Go to **[console.anthropic.com](https://console.anthropic.com)**.
2. Sign up and add a payment method.
3. Go to **API Keys**, then **Create Key**.
4. Copy the key (it also starts with `sk-`).

**Cost:**

| What you pay | Price |
|---|---|
| Input | ~$3.00 per 1 million tokens |
| Output | ~$15.00 per 1 million tokens |

*These are API rates and they add up quickly. Prices change, so check the Anthropic site.*

### 7c. Google Gemini: Optional Fallback

**Gemini** (by Google) has a free tier, which makes it a useful fallback.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers light usage. Paid rates apply above it.

---

## 8. Install Mullet

The installer uses **uv** or **pipx** (whichever you have) to install Mullet from GitHub as a tool of its own. It needs `git` and Python 3.10 to 3.12, which you set up above, and it stops with an error if you have neither uv nor pipx.

Check for them:

```bash
uv --version
pipx --version
```

If both say `command not found`, install one. uv is the simplest:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal. (On Ubuntu 22.04 or later, `sudo apt install -y pipx` also works.)

Now install Mullet:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

> **Do not run `pip install mullet`.** The package of that name on PyPI is someone else's project, not this one.

To update Mullet later, run the same command again.

**Optional extras:** Mullet's Gemini, Bedrock, Postgres and GPU support are extras. Add them with `MULLET_EXTRAS`:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | MULLET_EXTRAS=gemini bash
```

The extras are `gemini`, `bedrock`, `postgres`, `gpu` and `all`. You do not need any of them for jobs.

**Verify it worked:**

```bash
mullet --version
```

You should see `mullet` followed by the installed version number.

**If you see `mullet: command not found`**, the folder the installer used is not on your PATH yet. Run the command for the tool the installer used, then open a **new terminal**:

```bash
uv tool update-shell    # if you have uv
pipx ensurepath         # if you have pipx
```

**Where Mullet keeps things.** Config is in `~/.config/mullet`, state in `~/.local/state/mullet`, data (pools, skills, jobs) in `~/.local/share/mullet` and caches in `~/.cache/mullet`. `mullet paths` prints them. If you have an older `~/.mullet` folder, it keeps working until you run `mullet migrate-state` (add `--dry-run` first to see what it would move).

> **Working on Mullet itself?** That is a different route. Clone the repo, make a virtualenv and run `pip install -e ".[dev]"` inside it. Ordinary users do not need this.

---

## 9. Set Your API Keys (Optional)

**Skip this section if you only want to run jobs.** Jobs need no keys.

If you got provider keys in Section 7, put them in Mullet's own config file, not in `~/.bashrc` or `~/.zshrc`. Mullet reads `~/.config/mullet/.env` (or `$MULLET_CONFIG_DIR/.env` if you set that variable). A project's own `.env` file is never read.

1. Create the folder and the file, and make the file private:

```bash
mkdir -p ~/.config/mullet
touch ~/.config/mullet/.env
chmod 600 ~/.config/mullet/.env
```

2. Open the file in **nano**, a simple text editor that runs inside the terminal:

```bash
nano ~/.config/mullet/.env
```

3. Add one line per key, **replacing the placeholder text with your actual keys**. Use plain `KEY=value` lines. Do not add `export` or quotes:

```
KIMI_API_KEY=sk-your-actual-kimi-key-here
OPENROUTER_API_KEY=your-actual-openrouter-key-here
GOOGLE_API_KEY=your-actual-gemini-key-here
```

Add only the keys you have. `GOOGLE_API_KEY` is the Gemini key. Do **not** add `ANTHROPIC_API_KEY` unless you want API billing (see [Section 7b](#7b-anthropic-optional-billed-per-token)).

4. Save and exit nano:
   - Press `Control + O` (the letter O, not zero). Nano asks "File Name to Write."
   - Press `Return` to confirm.
   - Press `Control + X` to exit.

An environment variable of the same name overrides the file. If a key seems to be ignored, check that you have not exported a different value in your shell.

**Verify a key is set without printing it:**

```bash
grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env
```

`1` means the line is there. `0` means it is not. Avoid `echo $KIMI_API_KEY`: it prints the key on screen, and anything printed there should be treated as exposed.

---

## 10. Run mullet doctor

Mullet comes with a health check that tells you whether everything is set up correctly. It is read-only and works offline.

Run:

```bash
mullet doctor
```

You'll see output like this (yours will differ):

```
Mullet v<version> - System Health Check

  ✓ Python               3.12.4
  ✓ Claude Code CLI      <version> (Claude Code)
  ✓ Git                  2.43.0
  ✓ Bubblewrap           bubblewrap 0.8.0
  ✓ Job sandbox          Self-test passed
  ✓ socat                Found
  · Local LLM            not detected
  · Kimi API             KIMI_API_KEY not set
  · Claude Code setup    Not set up: run mullet setup-claude
  ...

21/24 checks passed
```

The checks that matter for jobs are Python, Claude Code, Git, Bubblewrap, Job sandbox and socat. The doctor also warns if `ANTHROPIC_API_KEY` is set.

**Here's what each symbol means:**

| Symbol | Color | Meaning |
|---|---|---|
| ✓ | Green | All good. This check passed. |
| ! | Yellow | Warning, not an error. Read the line. |
| · | Cyan | Informational. Something isn't set up yet, but it's okay to skip for now. |
| ✗ | Red | Problem that needs fixing before Mullet works properly. |

**If Bubblewrap, Job sandbox or socat is red**, install them. Jobs run inside this sandbox and refuse to start without it:

```bash
sudo apt install bubblewrap socat
```

> **Fedora / RHEL:** `sudo dnf install -y bubblewrap socat`
>
> **Arch:** `sudo pacman -S bubblewrap socat`

**Yellow and cyan items are optional.** If you see red, jump to the [Troubleshooting](#16-troubleshooting) section.

`mullet doctor --full` adds provider probes (it contacts the providers) and a GitHub check for a newer Mullet.

---

## 11. Set Up a Local LLM (Optional)

A **local LLM** is an AI model that runs entirely on your own Linux machine: no internet required, no API costs, and your code stays private. Mullet's delegation tools can send simple questions to your local model for free. **You can skip this section.** Jobs do not need a local model.

You have two options. **Pick one.**

> **RAM and GPU note:** A 7B model needs about 5 to 8 GB of free memory on top of everything else you run. **If your machine has 10 GB of RAM or less, use a 2B to 3B model instead** (for example `qwen2.5-coder:3b` in Ollama, or a 3B Qwen2.5-Coder in LM Studio), or skip local models. If you have a dedicated NVIDIA or AMD GPU with **8 GB or more of VRAM**, you can run larger models for better quality. Check your GPU with `nvidia-smi` (NVIDIA) or `rocminfo` (AMD). The steps below use the 7B model.

### Option A: LM Studio (Recommended for GUI lovers)

**LM Studio** is a friendly desktop app for running local AI models. On Linux, it comes as an **AppImage**, a single file that contains the entire application.

1. **Download LM Studio:**
   - Go to **[lmstudio.ai](https://lmstudio.ai)** in your web browser.
   - Click **Download for Linux**.
   - Save the `.AppImage` file to your Downloads folder.

2. **Make it executable:**
   - Open your terminal and run:

   ```bash
   chmod +x ~/Downloads/LM_Studio-*.AppImage
   ```

3. **Run LM Studio:**

   ```bash
   ~/Downloads/LM_Studio-*.AppImage
   ```

   > **If the AppImage won't run** and you see an error about `FUSE`, install the required library:
   >
   > **Ubuntu / Debian:** `sudo apt install -y libfuse2`
   >
   > **Fedora / RHEL:** `sudo dnf install -y fuse`
   >
   > **Arch:** `sudo pacman -S fuse2`

   > **If LM Studio crashes on launch** with a sandbox error, try running it with the `--no-sandbox` flag:
   >
   > ```bash
   > ~/Downloads/LM_Studio-*.AppImage --no-sandbox
   > ```

4. **Download a model:**
   - In LM Studio, click the **search icon** on the left sidebar (it looks like a magnifying glass).
   - In the search bar at the top, type: **`Qwen2.5-Coder-7B-Instruct`**
   - Click the model name in the results, then click **Download**.
   - This downloads about **4 GB** of data. Grab a coffee. It may take 10 to 20 minutes depending on your internet.
   - **What this model is:** A 7-billion-parameter AI trained specifically for coding tasks. It runs on most machines with more than 10 GB of RAM and can handle simple coding questions.

5. **Load the model:**
   - Once downloaded, click the **chat icon** on the left sidebar.
   - At the top of the chat panel, click the model dropdown and select **Qwen2.5-Coder-7B-Instruct**.
   - Click the **Load** button.

6. **Start the local server:**
   - Look for the **server toggle switch** in the top bar of LM Studio (it says "Server" with an on/off switch).
   - **Click the toggle to turn it ON.**
   - The default port is **1234**. You don't need to change this.

7. **Verify:**
   - Go back to your terminal and run:

   ```bash
   mullet doctor
   ```

   - You should now see a green **✓ Local LLM** line.

**Where to look:**
- **Search bar:** Top of the search panel (magnifying glass icon).
- **Server toggle:** Top center of the main window, labeled "Server."

### Option B: Ollama (Lighter, terminal-based)

**Ollama** is a lightweight tool that runs models from the command line. It's great if you prefer keeping everything in the terminal.

Install it with the official install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> **GPU Acceleration Callout:** If you have an **NVIDIA GPU**, Ollama automatically detects CUDA and uses it for much faster responses. Verify your GPU is recognized by running `nvidia-smi`. If you see your GPU listed with driver info, you're good to go.
>
> **AMD users:** Ollama also supports AMD ROCm on Linux. If you have an AMD GPU with ROCm drivers installed, Ollama will use them automatically.

Download the same coding model:

```bash
ollama pull qwen2.5-coder:7b
```

This downloads about **4 GB**. Once it finishes, start the server:

```bash
ollama serve
```

This runs in the foreground. **Leave this terminal window open.** If you close it, the local model stops.

*(To run it in the background later, you can set it up as a systemd service with `sudo systemctl enable --now ollama`, but that's outside the scope of this guide.)*

**Verify:**

Open a *new* terminal window and run:

```bash
mullet doctor
```

You should see a green **✓ Local LLM** line. Ollama runs on port **11434**, and Mullet detects it automatically.

---

## 12. Wire Mullet into Claude Code

One command registers Mullet's MCP server and skills with Claude Code. Run it with `--dry-run` first to see exactly what it will change:

```bash
mullet setup-claude --dry-run
```

Then run it for real:

```bash
mullet setup-claude
```

**What this does:**
- Registers the **MCP server** for every project (`claude mcp add --scope user mullet`), so Claude Code can call Mullet's tools
- Installs the `mullet-job` and `pool-improve` **skills** into `~/.claude/skills/`
- Adds **hooks** only if you ask: `mullet setup-claude --hooks list` describes them, and `--hooks name1,name2` adds the ones you pick
- Sets Claude Code's status line to Mullet's HUD only if you pass `--statusline` and none is set
- Never edits your `CLAUDE.md`. `mullet setup-claude --uninstall` removes exactly what it added

**Important:** After this command finishes, you must **fully quit and reopen Claude Code**.

- Press `Control + C` to exit Claude Code if it's running.
- Then type `claude` again in the terminal to restart it from scratch.

---

## 13. Test That Everything Works

Let's run a few quick tests. Tests 1 and 2 need no API keys.

### Test 1: Job commands work

List your jobs. A new install has none:

```bash
mullet job list
```

**Expected output:**

```
No jobs yet. Start one with: mullet job plan "<goal>"
```

### Test 2: Mullet tools appear in Claude Code

1. Open Claude Code in any project folder:

```bash
claude
```

2. Type `/mcp`. You should see `mullet` in the list of servers, connected.

If it is missing, or Claude says it doesn't know Mullet's tools, see [Troubleshooting](#16-troubleshooting).

### Test 3: See routing before spending money (optional)

Mullet can show you what it *would* do without making an LLM call:

```bash
mullet explain "refactor the authentication module"
```

**Expected output** (trimmed):

```
Routing Analysis: "refactor the authentication module"

  Simple message: No
  Complexity score: 80.0/100
  Routing tier: ADVANCED
  ...
  Provider: Premium cloud (Sonnet/Opus)
  Estimated cost: $0.0210
```

This tells you how hard Mullet thinks the request is, which tier it would send it to and about how much it would cost. The exact numbers depend on your version and settings. No LLM call is made during `explain`, so this test is free.

### Test 4: Delegation works (optional, needs a key or local model)

If you set up a provider key or a local model, ask Mullet a simple question from the terminal:

```bash
mullet "what Python version am I running?"
```

You should get an answer within a few seconds, from your local model or from a provider whose key you added. Inside Claude Code you can ask for the same thing with: `Use mullet_run to summarise what this project does.` Claude should show a tool call to a Mullet tool.

### Test 5: Run a small job

Jobs run with your own logged-in Claude Code, so this uses your Claude allowance. Go to a git repository you don't mind experimenting in and plan a small job:

```bash
cd ~/path/to/your/repo
mullet job plan "add a unit test for the parse_date function"
```

Mullet shows you the task tree and the checks it will run. Nothing has run yet. Read it, then approve and start it:

```bash
mullet job approve <job-id>
mullet job run <job-id>
```

The job runs in the background in a private clone. Follow it with:

```bash
mullet job status <job-id>
mullet job tail -f <job-id>
```

The other commands are `show`, `stop` (the job is paused, not lost), `resume`, `list` and `discard`. When it finishes, read the evidence report:

```bash
mullet job report <job-id>
```

If you are happy with it, copy the job's branch into your repo as a new branch:

```bash
mullet job merge <job-id>
```

`merge` never pushes. Your own checkout stays untouched until you run it. Jobs refuse to start on macOS for now, because the sandbox is Linux-only.

---

## 14. Knowledge Pools (Optional)

**Knowledge Pools** let Mullet index your codebase so Claude can search it intelligently instead of pasting entire files into the conversation.

If you're feeling overwhelmed, **skip this for now.** You can come back to it later. Pools live in the folder printed by `mullet paths pools`.

**Quick flow:**

1. In your project folder, run:

```bash
mullet init
```

This writes a `mullet.yaml`, some `.claude/` starter files and a pool named after your folder. Use `mullet init --no-claude` to skip the `.claude/` files. It skips any file that already exists.

2. Index your code, using your folder name as the pool name:

```bash
mullet reindex --pool myproject
```

3. Inside Claude Code, ask:

```
Use trawl_search_pool to find where the user authentication is handled.
```

Claude will use Mullet's semantic search to find the right files without you having to tell it where to look.

For full details, see the **[SKILLS.md](SKILLS.md)** document.

---

## 15. Pro Tip: Planning with Opus

Before starting any big coding task in Claude Code, try this command:

```
/model opusplan
```

Type it in the Claude Code chat window. This switches Claude to **Opus** just for the planning phase, so you get a thorough plan before any code is written. Once the plan is approved, Claude goes back to its normal model for the implementation. If you set up provider keys, you can ask Claude to hand simple, well-defined pieces of work to Mullet's delegation tools (Kimi or a local model). For a large piece of work, plan it as a job (`mullet job plan`) instead.

**When to use it:**
- Starting a new feature from scratch
- Tackling a complex bug you don't fully understand yet
- Refactoring a large section of code
- Any task where getting the approach right matters more than speed



---

## 16. Troubleshooting

Something not working? You're not alone. Here's a quick lookup table.

| Problem | Fix |
|---|---|
| `mullet: command not found` after install | The folder the installer used isn't in your PATH yet. Run `uv tool update-shell` (if you installed with uv) or `pipx ensurepath` (if pipx), then open a new terminal. |
| `AppImage won't run` | Install FUSE support:<br>**Ubuntu / Debian:** `sudo apt install -y libfuse2`<br>**Fedora / RHEL:** `sudo dnf install -y fuse`<br>**Arch:** `sudo pacman -S fuse2` |
| LM Studio crashes on launch | Try the `--no-sandbox` flag:<br>`~/Downloads/LM_Studio-*.AppImage --no-sandbox`<br>Also ensure `libfuse2` is installed. |
| Ollama not detected | Make sure `ollama serve` is running in another terminal. Verify the port with:<br>`curl http://localhost:11434`<br>You should see `Ollama is running`. |
| The installer says it needs uv or pipx | Install one, then run the installer again. uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh`. pipx on Ubuntu 22.04 or later: `sudo apt install -y pipx`. |
| npm/node not found after nvm install | Close and reopen your terminal. nvm needs a fresh shell to activate. Type `exit`, open a new terminal, and try again. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Control + C`) and reopen it by typing `claude`. Simply closing the window is not enough. |
| `mullet doctor` shows no providers | Providers are optional and only matter for delegation. Check that your keys are in `~/.config/mullet/.env` as plain `KEY=value` lines (no `export`). Run `grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env`: `0` means the line is missing. See [Section 9](#9-set-your-api-keys-optional). Keys in `~/.bashrc` still work as environment variables, but the `.env` file is the supported place. |
| Ollama slow without GPU | Check `nvidia-smi` to verify your NVIDIA GPU is detected. If it's not listed, install the NVIDIA drivers for your distro. For AMD, ensure ROCm is installed. CPU-only mode works but is slower. |
| Kimi key rejected | Make sure the line is `KIMI_API_KEY=sk-...` with no quotes or spaces. Check that you have credits at **[platform.moonshot.cn](https://platform.moonshot.cn)**. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong (often a missing Python package). |
| A job refuses to start with a sandbox error | Run `mullet doctor`. Install what it marks red: `sudo apt install bubblewrap socat`. |
| Jobs are billed to the API, not my subscription | `ANTHROPIC_API_KEY` is set in your shell or in `~/.config/mullet/.env`. Remove it for jobs. `mullet doctor` warns about it. |

Still stuck? Run the full diagnostic (it contacts the providers and checks GitHub for a newer Mullet) and share the output. Check it for keys before you paste it anywhere:

```bash
mullet doctor --full
```

---

## 17. Next Steps

You're all set. Mullet is installed and wired into Claude Code.

**Where to go from here:**

- **Quick reference for daily commands:** See **[QUICKSTART.md](QUICKSTART.md)**
- **How jobs are planned, gated and merged:** See **[MULLET2.md](MULLET2.md)**
- **How Mullet routes requests and chooses providers:** See **[ARCHITECTURE.md](ARCHITECTURE.md)**
- **Full installation reference (including Linux):** See **[INSTALL.md](INSTALL.md)**
- **Something broken?** Run `mullet doctor --full` and paste the output into a GitHub issue.
