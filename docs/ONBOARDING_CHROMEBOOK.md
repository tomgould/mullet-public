# Mullet on Chromebook — Complete Setup Guide

> **Mullet is a smart router for AI coding assistants.** Instead of sending every question to expensive AI models, Mullet automatically picks the cheapest provider that can handle the job — from a free local model on your Chromebook to cloud APIs like Kimi, Gemini, or Anthropic. By the end of this guide, you'll have Mullet running inside Claude Code, saving you money on every coding task while keeping responses fast and high-quality.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Enable Linux on Your Chromebook](#2-enable-linux-on-your-chromebook)
- [3. Open the Linux Terminal](#3-open-the-linux-terminal)
- [4. Install Python 3.10+](#4-install-python-310)
- [5. Install Git](#5-install-git)
- [6. Install Node.js via nvm](#6-install-nodejs-via-nvm)
- [7. Install Claude Code](#7-install-claude-code)
- [8. Get Your API Keys](#8-get-your-api-keys)
  - [8a. Kimi K2 — Recommended Primary Provider](#8a-kimi-k2--recommended-primary-provider)
  - [8b. Anthropic — Optional Orchestrator](#8b-anthropic--optional-orchestrator)
  - [8c. Google Gemini — Optional Fallback](#8c-google-gemini--optional-fallback)
- [9. Install Mullet](#9-install-mullet)
- [10. Set Your API Keys](#10-set-your-api-keys)
- [11. Run mullet doctor](#11-run-mullet-doctor)
- [12. Set Up Ollama](#12-set-up-ollama)
- [13. Wire Mullet into Claude Code](#13-wire-mullet-into-claude-code)
- [14. Test That Everything Works](#14-test-that-everything-works)
- [15. Pro Tip: Planning with Opus](#15-pro-tip-planning-with-opus)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Next Steps](#17-next-steps)

---

## 1. Before You Start

Here's what you'll need. Don't worry — everything below is explained step by step.

- **Chrome OS 105 or later.** To check: click the **time** in the bottom-right corner of your screen, then click the **Settings gear icon**. Scroll down to **About Chrome OS** and look at the version number. If it's 105 or higher, you're good to go.
- **At least 4 GB of RAM** and **20 GB of free disk space** inside Linux. If your Chromebook has **8 GB of RAM or more**, you'll be able to run a free local AI model too. If you have **4 GB of RAM**, you'll use **Kimi** as your primary provider (it's only about **$0.60 per million tokens** and needs zero RAM — see the note in [Section 12](#12-set-up-ollama)).
- **30–45 minutes** the first time you run through this. (The next time will take two minutes. You're doing great.)
- **A credit card handy** for API keys. Kimi, our recommended primary provider, costs roughly **$1 to get started**.

---

## 2. Enable Linux on Your Chromebook

Chrome OS keeps its regular apps and files separate from Linux tools. Before you can install Mullet, you need to turn on a special Linux container called **Crostini**. This is a one-time setup.

**Step by step:**

1. Click the **time** in the bottom-right corner, then click the **Settings gear icon**.
2. In the left sidebar, scroll down and click **Advanced** to expand it.
3. Click **Developers**.
4. Click **Linux development environment**.
5. Click **Turn On**.
6. A setup window will appear. Choose a username (or leave the default) and set a disk size of at least **20 GB**. Click **Install**.

**What happens next:**

- Chrome OS downloads about **500 MB** of Linux files. This takes **5–10 minutes** depending on your internet speed.
- Your Chromebook may restart. That's normal.
- When it's done, a new app called **Terminal** appears in your app launcher.

**If you don't see "Linux development environment":**

- Make sure your Chrome OS is up to date. Go to **Settings → About Chrome OS → Check for updates**. Some older Chromebooks don't support Linux — but most made after 2019 do.

---

## 3. Open the Linux Terminal

After Linux is enabled, you need to open the **Linux Terminal** app — not the Chrome browser's developer console.

**How to find it:**

- Click the **Launcher** (the circle icon in the bottom-left corner of your screen, or press the **Search** key on your keyboard).
- Type **Terminal**.
- Click the app called **Terminal** with a black screen icon.

**What it looks like:** A plain black window with a blinking cursor and text like `penguin@localhost:~$`. This is your Linux command line — the same kind developers use on servers around the world.

**Tip:** You can right-click the Terminal icon in the shelf (the bar at the bottom) and choose **Pin** to keep it there for easy access.

---

## 4. Install Python 3.10+

Mullet is written in **Python**, a popular programming language. The Linux container usually comes with Python already installed, but let's make sure it's new enough.

In Terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x` or higher** (for example, `Python 3.11.2`), you're good — skip to [Section 5](#5-install-git).

**If you see `command not found` or a version below 3.10**, run:

```bash
sudo apt update && sudo apt install -y python3 python3-pip
```

**What this does:** `sudo` means "do this as an administrator." `apt` is the tool Linux uses to download and install software. The `-y` flag automatically says "yes" to any prompts. This command updates the software list and installs Python in one go.

You may be asked for your Linux password (the one you set during Linux setup). Type it — you won't see any dots on screen for security reasons — and press `Return`.

**Verify it worked:**

```bash
python3 --version
```

You should now see `Python 3.10.x` or higher.

---

## 5. Install Git

**Git** is a tool for downloading and updating code from the internet. We'll use it to fetch Mullet.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set — skip to [Section 6](#6-install-nodejs-via-nvm).

**If you see `command not found`**, install Git via `apt`:

```bash
sudo apt update && sudo apt install -y git
```

**Verify it worked:**

```bash
git --version
```

---

## 6. Install Node.js via nvm

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

This downloads and installs nvm. **Close the Terminal window completely and reopen it** so nvm activates. (Click the X on the Terminal window, then reopen Terminal from the launcher.)

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

## 7. Install Claude Code

**Claude Code** is Anthropic's AI coding assistant that runs in your Terminal. Mullet plugs directly into it to route your questions to the cheapest, fastest AI model for each task.

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

## 8. Get Your API Keys

An **API key** is like a password that lets Mullet talk to AI services on your behalf. You'll add these keys to your Chromebook once, and Mullet will use them automatically.

You only **need** the Kimi key to get started. The others are optional but helpful.

### 8a. Kimi K2 — Recommended Primary Provider

**Kimi** (by Moonshot AI) is Mullet's default workhorse — fast, capable, and affordable.

1. Open your web browser and go to **[platform.moonshot.cn](https://platform.moonshot.cn)**.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add **$5** (about the cost of a coffee). This lasts a surprisingly long time.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-chromebook` and click **Create**.
6. **Copy the key immediately** — it looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal. That's how you know it's a Kimi key.
7. Paste it somewhere safe (like Google Keep or a text file in Google Drive) for now. We'll add it to your Chromebook in [Section 10](#10-set-your-api-keys).

**Cost:**

| What you pay | Price |
|---|---|
| Input (what you send) | ~$0.60 per 1 million tokens |
| Output (what you receive) | ~$2.50 per 1 million tokens |

*One million tokens is roughly 750,000 words — about three full novels. For everyday coding questions, you'll spend pennies.*

### 8b. Anthropic — Optional (Orchestrator)

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

### 8c. Google Gemini — Optional Fallback

**Gemini** (by Google) offers a generous free tier, making it a great safety net.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers most casual usage. If you exceed it, paid rates are very low.

---

## 9. Install Mullet

Now for the main event. These commands download Mullet from the internet and set it up on your Chromebook.

In Terminal, run these commands one at a time:

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .
```

**What `-e .` means:** The `-e` stands for "editable." It installs Mullet in a special way so that if the code is updated later, you get the changes instantly without reinstalling. The `.` means "install the project in this folder." Think of it like creating a shortcut instead of making a copy.

**Make sure your Chromebook can find Mullet** — run this now:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
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

## 10. Set Your API Keys

Now we'll tell your Chromebook to remember your API keys so Mullet can use them automatically.

We'll edit a hidden file called **`.bashrc`** using **nano**, a simple text editor that runs inside Terminal. (We use nano instead of other editors because it's friendly for beginners.)

1. Open nano:

```bash
nano ~/.bashrc
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
source ~/.bashrc
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

**Finding your Linux files:** Your `.bashrc` lives in the Linux part of your Chromebook. If you want to find it using the regular Chrome OS Files app, open Files and click **Linux files** in the left sidebar. You'll see your home folder there.

---

## 11. Run mullet doctor

Mullet comes with a built-in health check that tells you whether everything is set up correctly.

Run:

```bash
mullet doctor
```

You'll see output like this:

```
✓ Python 3.11 found
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

## 12. Set Up Ollama

A **local LLM** is an AI model that runs entirely on your own Chromebook — no internet required, no API costs, and your code stays private. Mullet can route simple questions to your local model for free, reserving your paid API keys for harder tasks.

On Chromebook, the only local LLM option is **Ollama**. (LM Studio is a Mac/Windows app and doesn't run on Linux.)

> **⚠️ RAM Warning:** Many Chromebooks come with only **4 GB of RAM**. Running a 7B model needs about **6 GB**. Before you continue, check your RAM:
>
> 1. Click the **time** in the bottom-right corner, then **Settings**.
> 2. Go to **About Chrome OS → Diagnostics → Memory**.
> 3. If you see **4 GB or less**, skip this section entirely. Use **Kimi** instead — it's only **$0.60 per million tokens**, needs zero RAM, and works perfectly with Mullet.
>
> If you have **8 GB or more**, go ahead!

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download a coding model:

```bash
ollama pull qwen2.5-coder:7b
```

This downloads about **4 GB**. Grab a coffee — it may take 10–20 minutes depending on your internet.

**What this model is:** A 7-billion-parameter AI trained specifically for coding tasks. It's small enough to run on many Chromebooks but smart enough to handle simple coding questions.

Once it finishes, start the server:

```bash
ollama serve
```

This runs in the foreground. **Leave this Terminal window open.** If you close it, the local model stops.

*(To run it in the background later, you can open a second Terminal window, but for now, just leave this one running.)*

**Verify:**

Open a *new* Terminal window (right-click the Terminal icon in the shelf and choose **New Window**) and run:

```bash
mullet doctor
```

You should see a green **✓ Local LLM** line. Ollama runs on port **11434**, and Mullet detects it automatically.

---

## 13. Wire Mullet into Claude Code

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
- Then type `claude` again in the Terminal to restart it from scratch.

---

## 14. Test That Everything Works

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

If Claude says it doesn't know that tool, see [Troubleshooting](#16-troubleshooting).

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
| Linux not appearing in Settings | Your Chromebook may need a Chrome OS update. Go to **Settings → About Chrome OS → Check for updates**. Install any available updates and restart. |
| `sudo apt update` fails | Check that your Chromebook is connected to WiFi. The Linux container shares your Chromebook's internet connection. |
| `python3: command not found` | Run `sudo apt update && sudo apt install -y python3 python3-pip`. |
| `mullet: command not found` after install | Add `~/.local/bin` to your PATH. Open `nano ~/.bashrc`, scroll to the bottom, and add:<br>`export PATH="$HOME/.local/bin:$PATH"`<br>Save (`Control + O`, `Return`, `Control + X`), then run `source ~/.bashrc`. |
| `npm: command not found` after nvm install | Close and reopen Terminal. nvm needs a fresh shell to activate. Click the X on the Terminal window, then reopen it from the launcher. |
| Ollama crashes or gets killed | Your Chromebook is low on RAM. Switch to Kimi instead: make sure `KIMI_API_KEY` is exported (run `echo $KIMI_API_KEY` to check), then skip Ollama and run `mullet doctor`. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Control + C`) and reopen it by typing `claude` in the Terminal. Simply closing the window is not enough. |
| Linux terminal closed mid-install | Just reopen it. Installations that completed are saved. You can safely rerun any command that was interrupted. |
| Kimi key rejected | Make sure your key starts with `sk-`. Check that you have credits at **[platform.moonshot.cn](https://platform.moonshot.cn)**. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong (often a missing Python package). |
| `pip install` fails with permissions error | Add the `--user` flag:<br>`pip install --user -e .` |

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
