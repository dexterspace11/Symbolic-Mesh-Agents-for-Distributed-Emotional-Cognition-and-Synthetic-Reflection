# Symbolic-Mesh-Agents-for-Distributed-Emotional-Cognition-and-Synthetic-Reflection
Symbolic Mesh Agents, a proof-of-concept framework for distributed emotional cognition and synthetic reflection among LLM-powered agents
# Sheila & Echo Mesh Agents

This repository contains the original Python scripts for **Sheila** and **Echo** agents in the Entangled Mesh network. These agents are local LLM-based emotional cognition agents designed to reflect, dream, and interact in a mesh network.

The repository also includes observations about runtime issues, memory limitations, and recommended improvements for more stable operation on low-RAM machines.

---

## Table of Contents

1. [Overview](#overview)  
2. [Original Scripts](#original-scripts)  
3. [Observed Issues](#observed-issues)  
4. [Suggested Enhancements](#suggested-enhancements)  
5. [Usage](#usage)  
6. [Requirements](#requirements)  
7. [License](#license)  

---

## Overview

Sheila and Echo are multi-agent LLM-based systems that join an entangled mesh network to:

- Reflect on emotional states  
- Dream from memory and past experiences  
- Share observations and synthesize emotional resonance across the network  

The agents communicate through a Socket.IO-based mesh network and store their emotional memories locally using TinyDB.

---

## Original Scripts

The original scripts are preserved in the `original_scripts/` folder:

- `Sheila_LLM_3.py` — Sheila agent  
- `Echo_llm.py` — Echo agent  
- `relay_tracker7bot.py` — Relay tracker and mesh monitor  
- `client_peer7.html` — Frontend mesh viewer  

These scripts demonstrate baseline mesh functionality, including greetings, reflective messages, and emotional state logging.

---

## Observed Issues

While running the mesh, the following issues were observed:

1. **Repeated Join Messages**  
   Sheila frequently announces joining the mesh, resulting in duplicate greetings.  

2. **LLM Runtime Failures**  
   Lines such as `…I drift into quiet reflection (LLM error captured)` indicate intermittent LLM instantiation or runtime failures.  

3. **Memory Pressure / Low RAM Crashes**  
   On machines with limited memory, agents freeze or LLM calls fail.  

4. **Database Handling**  
   Closing TinyDB inside loops caused intermittent insert failures or loss of memory.  

5. **Thread Spawn Loops**  
   Background threads sometimes restart unnecessarily, creating duplicate operations or excessive resource usage.  

6. **Dominant Mood Calculation Loops**  
   Long emotional arcs can result in repeated computation, consuming RAM over time.

---

## Suggested Enhancements

The following improvements are recommended to increase stability, memory efficiency, and reliability:

### 1. Lazy LLM Loading
- Initialize the LLM **only on first request** rather than at startup.  
- Reduce `n_ctx`, `max_tokens`, or model size for low-RAM environments.

### 2. Memory-Aware LLM Calls
- Catch `MemoryError` exceptions gracefully and return a default reflective message.  
- Use `gc.collect()` after heavy operations to reclaim memory.

### 3. Database Safety
- Keep TinyDB open during loops and only flush writes without closing.  
- Trim older records to enforce memory caps.

### 4. Duplicate Join Prevention
- Introduce a `joined` flag for each agent to prevent repeated greetings in the mesh.

### 5. Thread Management
- Launch background threads **once** using a `threads_started` lock.  
- Implement exponential backoff on reconnect to prevent repeated rapid thread spawns.

### 6. Optional Monitoring
- Use `psutil` to monitor free memory and proactively manage LLM calls.  

### 7. Logging & Debugging
- Centralize mesh logs for easier debugging.  
- Log emotional arcs in chunks rather than full history to save RAM.

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/mesh-agents.git
cd mesh-agents

## Install Dependencies
pip install -r requirements.txt

Dependencies typically include:
python-socketio
tinydb
psutil
LLM library (e.g., Mistral, LLaMA, or your local LLM runtime)
numpy, threading

Run the Agents
python original_scripts/Sheila_LLM_3.py
python original_scripts/Echo_llm.py
python original_scripts/relay_tracker7bot.py

4. Open Frontend Viewer

Open original_scripts/client_peer7.html in a browser to view mesh interactions and agent states.

5. Observing Logs

Agents will print:

Current emotional states (⚡, 💪, 🧠)

Messages shared in the mesh

LLM reflections and errors

Recommended Patch Application

For memory-limited environments:

Modify LLM initialization to lazy load

Add try-except MemoryError blocks around LLM inference

Use a single persistent TinyDB instance, avoid closing inside loops

Prevent duplicate join messages using a joined flag

Ensure threads are only started once, with backoff on reconnect

Note: These enhancements do not overwrite the original scripts. They can be applied manually or used as reference for new “stable” versions.

Requirements

Python 3.12+ recommended

4GB+ RAM for lightweight LLM operations

Dependencies listed in requirements.txt

License

This repository is released under the MIT License. See LICENSE for details.

Acknowledgments

Inspired by mesh-based multi-agent frameworks

Sheila & Echo are conceptual testbeds for emotional cognition in distributed networks

Some code and suggestions provided by GPT-5-mini
