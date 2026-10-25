# Mullet on Chromebook: Complete Setup Guide

> **Mullet runs large coding jobs for you and checks every step itself.** You approve a task tree, each step runs with your own logged-in Claude Code in a sandboxed private clone, and Mullet's own checks decide whether a step passed. You get back a branch and an evidence report. Your own files are untouched until you merge. Jobs need no API keys. Mullet can also send smaller tasks to cheaper models (Kimi, OpenRouter, Gemini or a local model) from inside Claude Code. That part is optional and needs keys. By the end of this guide, you'll have Mullet installed, wired into Claude Code, and ready to plan your first job.

## Table of Contents

- [1. Before You Start](#1-before-you-start)
- [2. Enable Linux on Your Chromebook](#2-enable-linux-on-your-chromebook)
- [3. Open the Linux Terminal](#3-open-the-linux-terminal)
- [4. Install Python 3.10+](#4-install-python-310)
- [5. Install Git](#5-install-git)
- [6. Install Node.js via nvm](#6-install-nodejs-via-nvm)
- [7. Install Claude Code](#7-install-claude-code)
- [8. Get Your API Keys (Optional)](#8-get-your-api-keys-optional)
  - [8a. Kimi K2: Recommended Cheap Provider](#8a-kimi-k2-recommended-cheap-provider)
  - [8b. Anthropic: Optional, Billed Per Token](#8b-anthropic-optional-billed-per-token)
  - [8c. Google Gemini: Optional Fallback](#8c-google-gemini-optional-fallback)
- [9. Install Mullet](#9-install-mullet)
- [10. Set Your API Keys (Optional)](#10-set-your-api-keys-optional)
- [11. Run mullet doctor](#11-run-mullet-doctor)
- [12. Set Up Ollama](#12-set-up-ollama)
- [13. Wire Mullet into Claude Code](#13-wire-mullet-into-claude-code)
- [14. Test That Everything Works](#14-test-that-everything-works)
- [15. Pro Tip: Plan First, Then Run a Job](#15-pro-tip-plan-first-then-run-a-job)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Next Steps](#17-next-steps)

---

## 1. Before You Start

Here's what you'll need. Don't worry, everything below is explained step by step.

- **Chrome OS 105 or later.** To check: click the **time** in the bottom-right corner of your screen, then click the **Settings gear icon**. Scroll down to **About Chrome OS** and look at the version number. If it's 105 or higher, you're good to go.
- **At least 4 GB of RAM** and **20 GB of free disk space** inside Linux. Jobs run Claude Code, which runs in the cloud, so they need very little RAM. A free local AI model is optional and needs more (see [Section 12](#12-set-up-ollama)). Chromebooks usually have 10 GB of RAM or less, so only a small model of 2B to 3B parameters is sensible.
- **A Claude subscription or Claude Console account** to log in to Claude Code. Jobs use your own login.
- **30 to 45 minutes** the first time you run through this. (The next time will take two minutes. You're doing great.)
- **A credit card handy only if you want the optional provider keys.** Kimi, the cheapest, costs roughly $1 to get started.

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

- Chrome OS downloads about **500 MB** of Linux files. This takes **5 to 10 minutes** depending on your internet speed.
- Your Chromebook may restart. That's normal.
- When it's done, a new app called **Terminal** appears in your app launcher.

**If you don't see "Linux development environment":**

- Make sure your Chrome OS is up to date. Go to **Settings > About Chrome OS > Check for updates**. Some older Chromebooks don't support Linux, but most made after 2019 do.

---

## 3. Open the Linux Terminal

After Linux is enabled, you need to open the **Linux Terminal** app, not the Chrome browser's developer console.

**How to find it:**

- Click the **Launcher** (the circle icon in the bottom-left corner of your screen, or press the **Search** key on your keyboard).
- Type **Terminal**.
- Click the app called **Terminal** with a black screen icon.

**What it looks like:** A plain black window with a blinking cursor and text like `penguin@localhost:~$`. This is your Linux command line, the same kind developers use on servers around the world.

**Tip:** You can right-click the Terminal icon in the shelf (the bar at the bottom) and choose **Pin** to keep it there for easy access.

---

## 4. Install Python 3.10+

Mullet is written in **Python**, a popular programming language. The Linux container usually comes with Python already installed, but let's make sure it's 3.10 or newer.

In Terminal, type:

```bash
python3 --version
```

**If you see `Python 3.10.x` or higher** (for example, `Python 3.11.2`), you're good, skip to [Section 5](#5-install-git).

**If you see `command not found` or a version below 3.10**, run:

```bash
sudo apt update && sudo apt install -y python3 python3-pip
```

**What this does:** `sudo` means "do this as an administrator." `apt` is the tool Linux uses to download and install software. The `-y` flag automatically says "yes" to any prompts. This command updates the software list and installs Python in one go.

You may be asked for your Linux password (the one you set during Linux setup). Type it and press `Return`. You won't see any dots on screen, which is normal.

**Verify it worked:**

```bash
python3 --version
```

You should now see `Python 3.10.x` or higher.

---

## 5. Install Git

**Git** is a tool for downloading and updating code from the internet. The Mullet installer uses it to fetch Mullet.

Check if you already have it:

```bash
git --version
```

**If you see something like `git version 2.39.x`**, you're all set, skip to [Section 6](#6-install-nodejs-via-nvm).

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

**Claude Code** is Anthropic's AI coding assistant that runs in your Terminal. Mullet plugs directly into it, and every Mullet job step runs through it.

Now install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
```

The `-g` flag means "install globally" so you can run it from anywhere.

**Verify it worked:**

```bash
claude --version
```

You should see a version number.

**Log in now.** Run `claude` once and follow the prompts to log in with your Claude account. Mullet jobs use this login, so they need no API key. Press `Control + C` twice to leave Claude Code when you're done.

---

## 8. Get Your API Keys (Optional)

**You can skip this whole section.** `mullet job` uses your own Claude Code login and needs no keys. Keys are only for Mullet's delegation tools, which send smaller tasks to cheaper models from inside Claude Code.

An **API key** is like a password that lets Mullet talk to an AI service on your behalf. If you want delegation, the Kimi key is the best one to start with. The others are optional.

### 8a. Kimi K2: Recommended Cheap Provider

**Kimi** (by Moonshot AI) is the cheapest capable provider Mullet can delegate to.

1. Open your web browser and go to **[platform.kimi.ai](https://platform.kimi.ai)**.
2. Click **Sign Up** and create an account with your email.
3. Once logged in, go to the **Billing** or **Top Up** section and add a small amount, such as $5. This lasts a long time for small tasks.
4. Go to the **API Keys** section and click **Create API Key**.
5. Give it a name like `mullet-chromebook` and click **Create**.
6. **Copy the key immediately.** You can't see it again. It looks like this: `sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - The `sk-` at the beginning is normal.
7. Paste it somewhere safe (like Google Keep or a text file in Google Drive) for now. We'll add it to your Chromebook in [Section 10](#10-set-your-api-keys-optional).

**Cost:** at the time of writing, roughly $1 per million input tokens and $4 per million output tokens. Check the provider's page for current prices.

*One million tokens is roughly 750,000 words, about three full novels. For everyday coding questions, you'll spend pennies.*

### 8b. Anthropic: Optional, Billed Per Token

**You almost certainly don't need this.** Mullet jobs and Claude Code use your Claude login, which is covered by your subscription. An `ANTHROPIC_API_KEY` is a separate, pay-per-token route.

**Warning:** if `ANTHROPIC_API_KEY` is set, `claude -p` (which Mullet jobs use) bills the API instead of your subscription. `mullet doctor` warns you when it is set. Only add it if you want that on purpose.

If you do:

1. Go to **[console.anthropic.com](https://console.anthropic.com)**.
2. Sign up and add a payment method.
3. Go to **API Keys** > **Create Key**.
4. Copy the key (it also starts with `sk-`).

Prices are on Anthropic's pricing page.

### 8c. Google Gemini: Optional Fallback

**Gemini** (by Google) offers a generous free tier, making it a useful safety net.

1. Go to **[aistudio.google.com](https://aistudio.google.com)**.
2. Sign in with your Google account.
3. Click **Get API Key** in the top menu.
4. Click **Create API Key** in a new project.
5. Copy the key (it looks like a long random string).

**Cost:** The free tier covers most casual usage. If you exceed it, paid rates are very low.

---

## 9. Install Mullet

Now for the main event. One command downloads Mullet from GitHub and installs it on your Chromebook. It needs **git** and **Python 3.10+** (you installed both above). The installer uses **uv** or **pipx**, whichever you have, and stops with a message if you have neither. See [INSTALL.md](INSTALL.md) for installing one of them.

In Terminal, run:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

**Never run `pip install mullet`.** The package of that name on PyPI is someone else's project, not this one.

To update Mullet later, run the same command again.

**Optional extras:** to install extra provider support, set `MULLET_EXTRAS` first. The choices are `gemini`, `bedrock`, `postgres`, `gpu` and `all`. For example:

```bash
MULLET_EXTRAS=gemini curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

**Verify it worked:**

```bash
mullet --version
```

You should see `mullet` followed by a version number.

**If `mullet: command not found`:** the installer puts `mullet` in a folder your shell may not search yet. Run `uv tool update-shell` (or `pipx ensurepath` if the installer used pipx), then close the Terminal window and open a new one.

**Install the job sandbox tools.** Jobs run inside a sandbox on Linux. It needs two small packages:

```bash
sudo apt install -y bubblewrap socat
```

A Chromebook's Linux container (Crostini) may not allow the sandbox to work. Mullet checks this for you in [Section 11](#11-run-mullet-doctor). If the sandbox self-test fails, a job refuses to run. There is no unsandboxed fallback. The other parts of Mullet (delegation tools, search) still work.

*Developer note:* working on Mullet itself is a different route: clone the repository, create a virtual environment, and run `pip install -e ".[dev]"`. Ordinary users should not do this.

---

## 10. Set Your API Keys (Optional)

**Skip this section if you only want jobs.** Keys are for delegation to Kimi, OpenRouter or Gemini.

Mullet reads keys from a file called **`.env`** in its config folder. It does not read your `.bashrc`. We'll create the file using **nano**, a simple text editor that runs inside Terminal.

1. Create the folder and open the file:

```bash
mkdir -p ~/.config/mullet
nano ~/.config/mullet/.env
```

2. Add the lines for the keys you have, as plain `NAME=value` lines. **Don't write `export`**, and replace the placeholder text with your actual keys:

```
KIMI_API_KEY=sk-your-actual-kimi-key-here
OPENROUTER_API_KEY=your-actual-openrouter-key-here
GOOGLE_API_KEY=your-actual-gemini-key-here
```

If you only got the Kimi key, add just that line. Add `ANTHROPIC_API_KEY=...` only if you want per-token billing (see [Section 8b](#8b-anthropic-optional-billed-per-token)).

3. Save and exit nano:
   - Press `Control + O` (that's the letter O, not zero). Nano will ask "File Name to Write."
   - Press `Return` to confirm.
   - Press `Control + X` to exit.

4. Make the file readable only by you:

```bash
chmod 600 ~/.config/mullet/.env
```

**Verify a key is set without printing it.** Don't use `echo`, because that shows your key on screen. Count the matching line instead:

```bash
grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env
```

`1` means the line is there. `0` means it is missing. Check the others the same way, changing the name.

**Good to know:**
- A key set in your shell environment wins over the file.
- If you set `MULLET_CONFIG_DIR`, Mullet reads `$MULLET_CONFIG_DIR/.env` instead.
- A project's own `.env` is never read.
- Gemini also accepts the name `GEMINI_API_KEY`.

**Finding your Linux files:** the `.env` file lives in the Linux part of your Chromebook. To see it in the Chrome OS Files app, open Files and click **Linux files** in the left sidebar. The `.config` folder is hidden, so turn on **Show hidden files** in the Files app menu.

---

## 11. Run mullet doctor

Mullet comes with a built-in health check that tells you whether everything is set up correctly. It is read-only and works offline. It checks Python, Claude Code, git, the job sandbox, your provider keys, a stray `ANTHROPIC_API_KEY`, and whether Mullet is wired into Claude Code.

Run:

```bash
mullet doctor
```

You'll see output like this (shortened):

```
✓ Python               3.11.2
✓ Claude Code CLI      (Claude Code)
✓ Git                  2.39.2
✓ Bubblewrap           bubblewrap 0.8.0
✓ Job sandbox          Self-test passed
✓ socat                Found
· Local LLM            not detected
· Google Gemini        GOOGLE_API_KEY not set
✓ Kimi API             KIMI_API_KEY set
· Claude Code setup    Not set up: run mullet setup-claude
```

The **Job sandbox** line is the important one for jobs. If it shows a warning, jobs will refuse to run on this machine (see [Troubleshooting](#16-troubleshooting)).

**Here's what each symbol means:**

| Symbol | Color | Meaning |
|---|---|---|
| ✓ | Green | All good! This check passed. |
| ! | Yellow | Warning, not an error. This is optional or needs attention later. |
| · | Cyan | Informational. Something isn't set up yet, but it's okay to skip for now. |
| ✗ | Red | Problem that needs fixing before Mullet works properly. |

At the bottom, you'll see a count like `20/24 checks passed`.

**Don't panic if some things are yellow or cyan.** Those items are optional. Red items are the ones to fix. If you see red, jump to the [Troubleshooting](#16-troubleshooting) section.

To also probe your providers and check GitHub for a newer Mullet, run `mullet doctor --full`. That one uses the network.

---

## 12. Set Up Ollama

**This section is optional.** A **local LLM** is an AI model that runs entirely on your own Chromebook, with no internet required, no API costs, and your code stays private. Mullet's delegation tools can send simple questions to it. Jobs don't need it.

On Chromebook, the only practical local LLM option is **Ollama**, which listens on port **11434**. (LM Studio, on port 1234, is a Mac/Windows app and doesn't run on Linux.)

> **RAM warning:** Most Chromebooks have **10 GB of RAM or less**. On a machine like that, use a small model of **2B to 3B parameters**. Don't try a 7B model or larger: it will be slow or get killed. Before you continue, check your RAM:
>
> 1. Click the **time** in the bottom-right corner, then **Settings**.
> 2. Go to **About Chrome OS > Diagnostics > Memory**.
> 3. If you see **4 GB or less**, skip this section. Use **Kimi** instead: it needs zero RAM.

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download a small coding model:

```bash
ollama pull qwen2.5-coder:3b
```

This downloads about **2 GB**. It may take 10 to 20 minutes depending on your internet.

**What this model is:** a 3-billion-parameter AI trained for coding. It handles simple questions only. We haven't tested it on a Chromebook, so treat it as a trial.

Once it finishes, start the server:

```bash
ollama serve
```

This runs in the foreground. **Leave this Terminal window open.** If you close it, the local model stops.

**Verify:**

Open a *new* Terminal window (right-click the Terminal icon in the shelf and choose **New Window**) and run:

```bash
mullet doctor
```

You should see a green **✓ Local LLM** line.

---

## 13. Wire Mullet into Claude Code

One command registers Mullet's MCP server and skills with Claude Code. Run it with `--dry-run` first to see exactly what it will change:

```bash
mullet setup-claude --dry-run
```

If you're happy with that, run it for real:

```bash
mullet setup-claude
```

**What this does:**
- Registers the **MCP server** for every project (`claude mcp add --scope user mullet -- <mullet> mcp`), so Claude Code can call Mullet's tools
- Installs the `mullet-job` and `pool-improve` **skills** into `~/.claude/skills/`
- Adds **hooks** only if you ask: `mullet setup-claude --hooks list` describes them, and `--hooks name1,name2` adds the ones you name
- Sets the status line only if you add `--statusline`
- Never edits your `CLAUDE.md`. `mullet setup-claude --uninstall` removes exactly what it added

**Important:** After this command finishes, you must **fully quit and reopen Claude Code**.

- Press `Control + C` to exit Claude Code if it's running.
- Then type `claude` again in the Terminal to restart it from scratch.

---

## 14. Test That Everything Works

Let's run a few quick tests to confirm Mullet is alive.

### Test 1: The sandbox passes

```bash
mullet doctor
```

Look for **Job sandbox** and **Claude Code setup**. Both should be green. If the sandbox line is not green, jobs won't run here (see [Troubleshooting](#16-troubleshooting)).

### Test 2: Plan a job

Go to a small project folder (a git repository) and ask Mullet for a plan. This only plans, so it changes nothing:

```bash
mullet job plan "add a README section that explains how to run the tests"
```

Mullet shows the task tree for your approval. Read it. Then list your jobs:

```bash
mullet job list
```

To throw the plan away, run `mullet job discard <id>` with the id from the list. To carry on, see [Section 15](#15-pro-tip-plan-first-then-run-a-job).

### Test 3: See routing before spending money

Mullet can show you how it would rate a request without making an API call:

```bash
mullet explain "refactor the authentication module"
```

**Expected output:** a routing analysis with a complexity score, a routing tier and the signals behind them. No LLM call is made, so this test is free.

### Test 4: MCP works inside Claude Code

1. Open Claude Code in any project folder:

```bash
claude
```

2. At the Claude prompt, type exactly:

```
Use mullet_status to show what Mullet is doing.
```

**Expected output:**

Claude should recognise the `mullet_status` tool and display a **tool call** in the chat. If that happens, Mullet is wired into Claude Code.

If Claude says it doesn't know that tool, see [Troubleshooting](#16-troubleshooting).

---

## 15. Pro Tip: Plan First, Then Run a Job

For a big task, plan before you run. A job always starts as a plan that you read and approve, so nothing runs until you say so.

```bash
mullet job plan "<what you want done>"   # create the plan
mullet job show <id>                     # read the task tree and the checks
mullet job approve <id>                  # say yes
mullet job run <id>                      # start it in the background
mullet job status <id>                   # see progress
mullet job tail <id> -f                  # follow events live
```

When it finishes:

```bash
mullet job report <id>                   # the evidence report
mullet job merge <id>                    # copy the finished branch into your repo
```

Other commands: `mullet job stop <id>` pauses a running job, `mullet job resume <id>` carries on, and `mullet job discard <id>` deletes the clone. Your checkout is untouched until you run `merge`.

Each step passes only when Mullet's own checks pass, not because the model says it is done. That is why the report is worth reading.

You can also ask Claude Code to do this for you: the `mullet-job` skill, installed in Section 13, covers it.

**Coming, not built yet:** choosing cheaper models for individual job steps, a benchmark command (`mullet job bench`), a guard that stops jobs overloading your machine, and job sandboxing on macOS.

**Tip:** Claude Code's `/model opusplan` command uses Opus for planning and a cheaper model for the work. It is separate from Mullet jobs.

---

## 16. Troubleshooting

Something not working? You're not alone. Here's a quick lookup table.

| Problem | Fix |
|---|---|
| Linux not appearing in Settings | Your Chromebook may need a Chrome OS update. Go to **Settings > About Chrome OS > Check for updates**. Install any available updates and restart. |
| `sudo apt update` fails | Check that your Chromebook is connected to WiFi. The Linux container shares your Chromebook's internet connection. |
| `python3: command not found` | Run `sudo apt update && sudo apt install -y python3 python3-pip`. |
| The installer says it needs uv or pipx | Install one of them (see [INSTALL.md](INSTALL.md)), then run the install command again. |
| `mullet: command not found` after install | Run `uv tool update-shell` (or `pipx ensurepath` if you used pipx), then close the Terminal window and open a new one. |
| `npm: command not found` after nvm install | Close and reopen Terminal. nvm needs a fresh shell to activate. Click the X on the Terminal window, then reopen it from the launcher. |
| `mullet doctor` shows the job sandbox failing, or a job refuses to run | Run `sudo apt install -y bubblewrap socat`, then `mullet doctor` again. If the self-test still fails, this Chromebook's Linux container does not allow the sandbox. There is no unsandboxed fallback, so jobs can't run here. Delegation tools and search still work. |
| `mullet doctor` warns about `ANTHROPIC_API_KEY` | It is set, so `claude -p` bills the API instead of your subscription. Remove the line from `~/.config/mullet/.env` and from your shell profile if you don't want that. |
| Mullet can't find a key I added | Keys go in `~/.config/mullet/.env` as `NAME=value` lines with no `export`. Not in `.bashrc`. Check with `grep -c '^KIMI_API_KEY=' ~/.config/mullet/.env`. |
| Ollama crashes or gets killed | Your Chromebook is low on RAM. Use a 2B to 3B model, or skip Ollama and use Kimi: check the key line is in `~/.config/mullet/.env`, then run `mullet doctor`. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude` again. Then **fully quit Claude Code** (`Control + C`) and reopen it by typing `claude` in the Terminal. Simply closing the window is not enough. |
| Linux terminal closed mid-install | Just reopen it. Installations that completed are saved. You can safely rerun any command that was interrupted. |
| Kimi key rejected | Check that you copied the whole key and that you have credits at **[platform.kimi.ai](https://platform.kimi.ai)**. |
| Claude Code says "mullet MCP server failed to start" | In Terminal, run `mullet mcp`. If you see an error, that error message tells you what's wrong. Running the installer again often fixes a missing Python package. |
| You have an older `~/.mullet` folder | It keeps working. To move it to the new locations, run `mullet migrate-state --dry-run` first, then `mullet migrate-state`. |

Still stuck? Run the full diagnostic and share the output:

```bash
mullet doctor --full
```

`mullet doctor` prints no key values, but read the output before you share it.

---

## 17. Next Steps

You're all set!

**Where Mullet keeps things:** `mullet paths` prints them.

| What | Folder |
|---|---|
| Config and keys | `~/.config/mullet` |
| State | `~/.local/state/mullet` |
| Data (pools, skills, jobs) | `~/.local/share/mullet` |
| Caches | `~/.cache/mullet` |

Search pools live in the folder that `mullet paths pools` prints. Set `TRAWL_POOLS_DIR` to move them. To rebuild a pool's index, run `mullet reindex --pool <name>`.

**Where to go from here:**

- **Quick reference for daily commands:** See **[QUICKSTART.md](QUICKSTART.md)**
- **How Mullet routes requests and chooses providers:** See **[ARCHITECTURE.md](ARCHITECTURE.md)**
- **Full installation reference (including Linux):** See **[INSTALL.md](INSTALL.md)**
- **Something broken?** Run `mullet doctor --full` and paste the output into a GitHub issue.

**Welcome to the Mullet family.** Happy coding.
