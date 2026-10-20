# Mullet on Linux — Complete Setup Guide

> **Mullet is a smart router for AI coding assistants.** Instead of sending every question to expensive AI models, Mullet automatically picks the cheapest provider that can handle the job — from a free local model on your Linux machine to cloud APIs like Kimi, Gemini, or Anthropic. By the end of this guide, you'll have Mullet running inside Claude Code, saving you money on every coding task while keeping responses fast and high-quality.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Check Your Shell](#2-check-your-shell)
- [3. Install Python 3.10+](#3-install-python-310)
- [4. Install Git](#4-install-git)
- [5. Install Node.js via nvm](#5-install-nodejs-via-nvm)
- [6. Install Claude Code](#6-install-claude-code)
- [7. Get Your API Keys](#7-get-your-api-keys)
  - [7a. Kimi K2 — Recommended Primary Provider](#7a-kimi-k2--recommended-primary-provider)
  - [7b. Anthropic — Optional Orchestrator](#7b-anthropic--optional-orchestrator)
  - [7c. Google Gemini — Optional Fallback](#7c-google-gemini--optional-fallback)
- [8. Install Mullet](#8-install-mullet)
- [9. Set Your API Keys](#9-set-your-api-keys)
- [10. Run mullet doctor](#10-run-mullet-doctor)
- [11. Set Up a Local LLM](#11-set-up-a-local-llm)
  - [Option A: LM Studio Recommended for GUI lovers](#option-a-lm-studio-recommended-for-gui-lovers)
  - [Option B: Ollama Lighter, terminal-based](#option-b-ollama-lighter-terminal-based)
- [12. Wire Mullet into Claude Code](#12-wire-mullet-into-claude-code)
- [13. Test That Everything Works](#13-test-that-everything-works)
- [14. Knowledge Pools Optional — Skip if Overwhelmed](#14-knowledge-pools-optional--skip-if-overwhelmed)
- [15. Pro Tip: Planning with Opus](#15-pro-tip-planning-with-opus)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Next Steps](#17-next-steps)

---

## 1. Before You Start

Here's what you'll need. Don't worry — everything below is explained step by step.

- **Ubuntu 20.04 or later, or an equivalent Linux distribution.** These instructions focus on **Ubuntu** and **Debian**, which are the most common starting points for newcomers. If you use Fedora, RHEL, or Arch, we've included quick callouts for you too.
- **At least 8 GB of RAM** and **20 GB of free disk space.** Local AI models need room to breathe.
- **A GPU is optional but great.** If you have an NVIDIA or AMD graphics card with 8 GB or more of video memory (VRAM), you can run larger local models for free and get faster responses. If not, no problem — Mullet will route you to affordable cloud providers.
- **30–45 minutes** the first time you run through this. (The next time will take two minutes. You're doing great.)
- **A credit card handy** for API keys. Kimi, our recommended primary provider, costs roughly **$1 to get started**.

---

## 2. Check Your Shell

Your **shell** is the program that interprets the commands you type into the terminal. On Ubuntu, the default is **bash**, but many Linux users prefer **zsh**. The shell you use determines which hidden file we edit to save your API keys and PATH settings.

In your terminal, run:

```bash
echo $SHELL
```

**If you see `/bin/bash` or `/usr/bin/bash`**, you use **bash**. We'll edit `~/.bashrc` throughout this guide.

**If you see `/bin/zsh` or `/usr/bin/zsh`**, you use **zsh**. We'll edit `~/.zshrc` instead.

**If you see something else** (like `/bin/fish`), we recommend switching to bash or zsh for this guide, or adapting the config file path to your shell.

> **Note for zsh users:** Whenever this guide says `~/.bashrc`, use `~/.zshrc` instead. Whenever it says `source ~/.bashrc`, run `source ~/.zshrc` instead.

---

## 3. Install Python 3.10+

Mullet is written in **Python**, a popular programming language. Most Linux distributions come with Python already installed, but let's make sure it's new enough.

In your terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x` or higher** (for example, `Python 3.12.4`), you're good — skip to [Section 4](#4-install-git).

**If you see `command not found` or a version below 3.10**, install Python via your package manager:

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y python3 python3-pip
```

> **Fedora / RHEL:** `sudo dnf install -y python3 python3-pip`
>
> **Arch:** `sudo pacman -S python python-pip`

You may be asked for your password. Type it — you won't see any dots on screen for security reasons — and press `Return`.

**Verify it worked:**

```bash
python3 --version
```

You should now see `Python 3.10.x` or higher.

---

## 4. Install Git

**Git** is a tool for downloading and updating code from the internet. We'll use it to fetch Mullet.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set — skip to [Section 5](#5-install-nodejs-via-nvm).

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

**Claude Code** is installed via **npm**, a tool that comes with **Node.js**. Node.js lets you run JavaScript programs, and npm installs them.

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

**Claude Code** is Anthropic's AI coding assistant that runs in your terminal. Mullet plugs directly into it to route your questions to the cheapest, fastest AI model for each task.

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

## 7. Get Your API Keys

An **API key** is like a password that lets Mullet talk to AI services on your behalf. You'll add these keys to your Linux machine once, and Mullet will use them automatically.

You only **need** the Kimi key to get started. The others are optional but helpful.

### 7a. Kimi K2 — Recommended Primary Provider

**Kimi** (by Moonshot AI) is Mullet's default workhorse — fast, capable, and affordable.

1. Open your web browser and go to **[platform.moonshot.cn](https://platform.moonshot.cn)**.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add **$5** (about the cost of a coffee). This lasts a surprisingly long time.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-linux` and click **Create**.
6. **Copy the key immediately** — it looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal. That's how you know it's a Kimi key.
7. Paste it somewhere safe (like a text file or password manager) for now. We'll add it to your Linux machine in [Section 9](#9-set-your-api-keys).

**Cost:**

| What you pay | Price |
|---|---|
| Input (what you send) | ~$0.60 per 1 million tokens |
| Output (what you receive) | ~$2.50 per 1 million tokens |

*One million tokens is roughly 750,000 words — about three full novels. For everyday coding questions, you'll spend pennies.*

### 7b. Anthropic — Optional (Orchestrator)

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

### 7c. Google Gemini — Optional Fallback

**Gemini** (by Google) offers a generous free tier, making it a great safety net.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers most casual usage. If you exceed it, paid rates are very low.

---

## 8. Install Mullet

Now for the main event. These commands download Mullet from the internet and set it up on your Linux machine.

In your terminal, run these commands one at a time:

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .
```

**What `-e .` means:** The `-e` stands for "editable." It installs Mullet in a special way so that if the code is updated later, you get the changes instantly without reinstalling. The `.` means "install the project in this folder." Think of it like creating a shortcut instead of making a copy.

**Make sure your system can find Mullet** — run this now:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

> **zsh users:** replace `~/.bashrc` with `~/.zshrc` in the commands above.

**Verify it worked:**

```bash
mullet --version
```

You should see something like:

```
mullet 0.x.x
```

---

## 9. Set Your API Keys

Now we'll tell your Linux machine to remember your API keys so Mullet can use them automatically.

We'll edit a hidden file using **nano**, a simple text editor that runs inside the terminal. (We use nano instead of other editors because it's friendly for beginners.)

1. Open nano:

**If you use bash:**
```bash
nano ~/.bashrc
```

**If you use zsh:**
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

5. Tell your terminal to reload the file:

**If you use bash:**
```bash
source ~/.bashrc
```

**If you use zsh:**
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

## 10. Run mullet doctor

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

**Don't panic if some things are yellow.** Yellow items are optional. Red items are the ones to fix. If you see red, jump to the [Troubleshooting](#16-troubleshooting) section for the fix.

---

## 11. Set Up a Local LLM

A **local LLM** is an AI model that runs entirely on your own Linux machine — no internet required, no API costs, and your code stays private. Mullet can route simple questions to your local model for free, reserving your paid API keys for harder tasks.

You have two options. **Pick one.**

> **GPU Note:** If you have a dedicated NVIDIA or AMD GPU with **8 GB or more of VRAM**, you can run larger models like 13B or 34B for even better quality. Check your GPU with `nvidia-smi` (NVIDIA) or `rocminfo` (AMD). If you only have integrated graphics or a small GPU, stick with the 7B model — it still works great.

### Option A: LM Studio (Recommended for GUI lovers)

**LM Studio** is a friendly desktop app for running local AI models. On Linux, it comes as an **AppImage** — a single file that contains the entire application.

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
   - This downloads about **4 GB** of data. Grab a coffee — it may take 10–20 minutes depending on your internet.
   - **What this model is:** A 7-billion-parameter AI trained specifically for coding tasks. It's small enough to run smoothly on most Linux machines but smart enough to handle simple coding questions.

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

**Screenshot callouts:**
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

- Press `Control + C` to exit Claude Code if it's running.
- Then type `claude` again in the terminal to restart it from scratch.

---

## 13. Test That Everything Works

Let's run three quick tests to confirm Mullet is alive and routing correctly.

### Test 1 — CLI works

Ask Mullet a simple question directly from the terminal:

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

If Claude says it doesn't know that tool, see [Troubleshooting](#16-troubleshooting).

---

## 14. Knowledge Pools (Optional — Skip if Overwhelmed)

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

## 16. Troubleshooting

Something not working? You're not alone. Here's a quick lookup table.

| Problem | Fix |
|---|---|
| `mullet: command not found` after install | Your `~/.local/bin` folder isn't in your PATH. Open your shell config (`nano ~/.bashrc` or `nano ~/.zshrc`), scroll to the bottom, and add:<br>`export PATH="$HOME/.local/bin:$PATH"`<br>Save (`Control + O`, `Return`, `Control + X`), then run `source ~/.bashrc` or `source ~/.zshrc`. |
| `AppImage won't run` | Install FUSE support:<br>**Ubuntu / Debian:** `sudo apt install -y libfuse2`<br>**Fedora / RHEL:** `sudo dnf install -y fuse`<br>**Arch:** `sudo pacman -S fuse2` |
| LM Studio crashes on launch | Try the `--no-sandbox` flag:<br>`~/Downloads/LM_Studio-*.AppImage --no-sandbox`<br>Also ensure `libfuse2` is installed. |
| Ollama not detected | Make sure `ollama serve` is running in another terminal. Verify the port with:<br>`curl http://localhost:11434`<br>You should see `Ollama is running`. |
| `pip: command not found` | Use `pip3` instead of `pip`, or install it with:<br>**Ubuntu / Debian:** `sudo apt install -y python3-pip` |
| npm/node not found after nvm install | Close and reopen your terminal. nvm needs a fresh shell to activate. Type `exit`, open a new terminal, and try again. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Control + C`) and reopen it by typing `claude`. Simply closing the window is not enough. |
| `mullet doctor` shows no providers | Check that your keys are exported. Run `echo $KIMI_API_KEY` — if it prints nothing, revisit [Section 9](#9-set-your-api-keys). |
| Ollama slow without GPU | Check `nvidia-smi` to verify your NVIDIA GPU is detected. If it's not listed, install the NVIDIA drivers for your distro. For AMD, ensure ROCm is installed. CPU-only mode works but is slower. |
| Kimi key rejected | Make sure your key starts with `sk-`. Check that you have credits at **[platform.moonshot.cn](https://platform.moonshot.cn)**. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong (often a missing Python package). |
| `python3 -m pip install` fails with permissions error | Add the `--user` flag:<br>`pip install --user -e .` |

Still stuck? Run the full diagnostic and share the output:

```bash
mullet doctor --full
```

---

## 17. Next Steps

You're all set! Mullet is now saving you money on every AI-assisted coding task.

**Where to go from here:**

- **Quick reference for daily commands:** See **[QUICKSTART.md](QUICKSTART.md)**
- **How Mullet routes requests and chooses providers:** See **[ARCHITECTURE.md](ARCHITECTURE.md)**
- **Full installation reference (including Linux):** See **[INSTALL.md](INSTALL.md)**
- **Something broken?** Run `mullet doctor --full` and paste the output into a GitHub issue.

**Welcome to the Mullet family.** Happy coding — and happy saving.
