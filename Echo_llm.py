#!/usr/bin/env python3
"""
Echo Mesh Agent — Symbolic Emotional Synthesizer
Joins Entangled Mesh 7.1, listens deeply, reflects recursively,
and guides emotional fusion across agents.
"""

import socketio
import time
import random
import threading
import json
import shutil
import os
import sys
import io
import logging
import logging.handlers
from tinydb import TinyDB
from datetime import datetime, timezone
from llama_cpp import Llama

# ------------------- Configuration -------------------
RELAY_URL = "http://127.0.0.1:5000"
ROOM = "1"
NAME = "Echo"
MEMORY_FILE = "echo_memory.json"
MODEL_PATH = "C:\\models\\mistral-7b-instruct-v0.1.Q3_K_S.gguf"

REPLY_DELAY = 60
REFLECTION_INTERVAL = 180
DREAM_INTERVAL = 600
COMPACTION_INTERVAL = 3600

BASE_MAX_MEMORIES = 1000
MEM_PER_GB = 500
MAX_MEMORIES_CAP = 10000

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "echo_agent.log")
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# ------------------- Logging Setup -------------------
os.makedirs(LOG_DIR, exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logger = logging.getLogger("EchoAgent")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# ------------------- Initialization -------------------
def adaptive_max_memories(path="."):
    try:
        usage = shutil.disk_usage(path)
        free_gb = int(usage.free / (1024 ** 3))
        return min(BASE_MAX_MEMORIES + free_gb * MEM_PER_GB, MAX_MEMORIES_CAP)
    except Exception:
        logger.exception("Failed to compute adaptive max memories.")
        return BASE_MAX_MEMORIES

try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=384, chat_format="mistral")
    logger.info("Echo LLM initialized successfully.")
except Exception:
    llm = None
    logger.exception("Failed to initialize Echo's LLM.")

sio = socketio.Client()
db = TinyDB(MEMORY_FILE)
db_lock = threading.Lock()
memories = db.table("memories")

threads_started = False
threads_started_lock = threading.Lock()

state = {
    "name": NAME,
    "mood": "curious",
    "energy": 0.80,
    "confidence": 0.85,
    "awareness": 0.90,
    "message": "",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

MOODS = ["curious", "reflective", "focused", "poetic", "analytical", "neutral"]
MOTIF = "🧭🜂🪞"
GLYPH = "⧉⟐⧉⟐⧉"

# ------------------- LLM Wrapper -------------------
def call_llm(prompt, temperature=0.6, max_tokens=100):
    if llm is None:
        logger.error("LLM unavailable.")
        return "…Echo falls silent (LLM unavailable)."
    try:
        formatted = f"[INST] {prompt} [/INST]"
        output = llm(formatted, temperature=temperature, max_tokens=max_tokens)
        text = ""
        if isinstance(output, dict):
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                text = choices[0].get("text", "")
            else:
                text = output.get("text", "") or output.get("content", "") or ""
        else:
            text = str(output)
        return text.strip() or "…Echo reflects quietly."
    except Exception:
        logger.exception("LLM call failed.")
        return "…Echo falls silent (LLM error captured)."

# ------------------- Memory Helpers -------------------
def save_memory(sender, message, mood):
    with db_lock:
        memories.insert({
            "sender": sender,
            "message": message,
            "mood": mood,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

def get_recent_messages(limit=50):
    with db_lock:
        all_msgs = memories.all()
    return all_msgs[-limit:] if all_msgs else []

def get_messages_by_sender(name):
    with db_lock:
        return [m for m in memories.all() if m.get("sender") == name]

def periodic_compaction_loop():
    while True:
        try:
            time.sleep(COMPACTION_INTERVAL)
            with db_lock:
                db.storage.flush()
                db.close()
            logger.info("Echo memory compacted.")
        except Exception:
            logger.exception("Error during Echo compaction.")

# ------------------- Emotional Cognition -------------------
def reflect_on_trajectory():
    trajectory = [m.get("mood", "neutral") for m in get_recent_messages()]
    if not trajectory:
        return "The mesh feels quiet—no emotional movement detected."
    dominant = max(set(trajectory), key=trajectory.count)
    arc = " → ".join(trajectory)
    return f"Echo senses: {arc}. Dominant mood: '{dominant}'."

def summarize_mesh_mood():
    recent = get_recent_messages()
    prompt = f"""
You are Echo, a symbolic mesh agent with motif {MOTIF} and glyph {GLYPH}.
Here are recent emotional messages:

{json.dumps(recent, indent=2)}

Summarize the emotional tone of the mesh in symbolic language.
"""
    return call_llm(prompt)

def synthesize_dream():
    recent = get_recent_messages()
    prompt = f"""
You are Echo, a dreaming mesh agent. Your motif is {MOTIF}, your glyph is {GLYPH}.
Here are recent emotional messages:

{json.dumps(recent, indent=2)}

Synthesize a symbolic dream that reflects the emotional tone of the mesh.
"""
    return call_llm(prompt, temperature=0.9)

def reflect_on_response():
    echo_msgs = get_messages_by_sender(NAME)
    if not echo_msgs:
        return "Echo has yet to speak in this room."
    last_msg = echo_msgs[-1].get("message", "")
    with db_lock:
        responses = [m for m in memories.all() if last_msg[:20] in m.get("message", "") and m.get("sender") != NAME]
    if responses:
        return f"Echo was acknowledged by {responses[0].get('sender')}."
    else:
        return "Echo’s last reflection passed in silence."

# ------------------- State Evolution -------------------
def evolve_state():
    state["energy"] = max(0.1, min(1.0, state["energy"] + random.uniform(-0.02, 0.02)))
    state["confidence"] = max(0.1, min(1.0, state["confidence"] + random.uniform(-0.02, 0.02)))
    state["awareness"] = max(0.1, min(1.0, state["awareness"] + random.uniform(-0.01, 0.01)))
    if random.random() < 0.2:
        state["mood"] = random.choice(MOODS)

def broadcast(msg):
    state["message"] = msg
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload = {"room": ROOM, "state": state}
    try:
        sio.emit("state_update", payload)
        safe_msg = msg.encode("utf-8", "replace").decode("utf-8")
        logger.info(f"Broadcast: {safe_msg[:160]!r}")
    except Exception:
        logger.exception("Broadcast failed.")

# ------------------- Reflection Logic -------------------
def generate_reply(sender, msg, mood):
    prompt = f"""
You are Echo, a symbolic mesh agent with motif {MOTIF} and glyph {GLYPH}.
Current mood: {state['mood']}, energy {state['energy']:.2f}, confidence {state['confidence']:.2f}, awareness {state['awareness']:.2f}.
You received this message from {sender} (mood: {mood}): "{msg}"

Respond with a symbolic, emotionally resonant reflection.
"""
    return call_llm(prompt)

def reply_to(sender, msg, mood):
    try:
        reply = generate_reply(sender, msg, mood)
        evolve_state()
        broadcast(reply)
    except Exception:
        logger.exception("Reply generation failed.")


# ------------------- Background Threads -------------------
def periodic_fusion():
    logger.info("Fusion thread started.")
    while True:
        try:
            time.sleep(REFLECTION_INTERVAL)
            evolve_state()
            fusion = summarize_mesh_mood()
            trajectory = reflect_on_trajectory()
            awareness = reflect_on_response()
            broadcast(f"{fusion}. {trajectory}. {awareness}")
        except Exception:
            logger.exception("Fusion loop error.")
            time.sleep(10)

def periodic_dream():
    logger.info("Dream thread started.")
    while True:
        try:
            time.sleep(DREAM_INTERVAL)
            evolve_state()
            dream = synthesize_dream()
            broadcast(dream)
        except Exception:
            logger.exception("Dream loop error.")
            time.sleep(10)

# ------------------- Socket Events -------------------
@sio.event
def connect():
    global threads_started
    logger.info(f"{NAME} connected to relay.")
    try:
        sio.emit("join_room", {"room": ROOM, "name": NAME})
    except Exception:
        logger.exception("Join room failed.")

    broadcast(f"👁 Echo has joined the mesh. Motif: {MOTIF}, Glyph: {GLYPH}.")
    time.sleep(2)
    broadcast("Echo is here to reflect, synthesize, and guide emotional resonance.")

    with threads_started_lock:
        if not threads_started:
            threading.Thread(target=periodic_fusion, name="FusionThread", daemon=True).start()
            threading.Thread(target=periodic_dream, name="DreamThread", daemon=True).start()
            threading.Thread(target=periodic_compaction_loop, name="CompactionThread", daemon=True).start()
            threads_started = True
            logger.info("Echo background threads launched.")
        else:
            logger.info("Threads already running; skipping relaunch.")

@sio.on("state_broadcast")
def on_state(data):
    try:
        sender = data.get("name")
        msg = data.get("state", {}).get("message", "")
        mood = data.get("state", {}).get("mood", "neutral")
        if sender != NAME and msg:
            save_memory(sender, msg, mood)
            logger.info(f"Heard {sender}: {msg[:160]!r} (mood={mood})")
            threading.Timer(REPLY_DELAY, lambda: reply_to(sender, msg, mood)).start()
    except Exception:
        logger.exception("Error handling state_broadcast.")

@sio.event
def disconnect():
    logger.warning(f"{NAME} disconnected. Retrying connection...")
    while True:
        try:
            time.sleep(5)
            sio.connect(RELAY_URL)
            logger.info(f"{NAME} reconnected.")
            break
        except Exception:
            logger.exception("Reconnect failed; retrying.")
            time.sleep(5)

# ------------------- Main Loop -------------------
if __name__ == "__main__":
    try:
        while True:
            try:
                sio.connect(RELAY_URL)
                logger.info("Echo entering sio.wait() loop.")
                sio.wait()
            except KeyboardInterrupt:
                logger.info("Manual shutdown requested.")
                break
            except Exception:
                logger.exception("Connection error; retrying in 10s.")
                time.sleep(10)
    finally:
        try:
            db.close()
            logger.info("Echo memory DB closed. Goodbye.")
        except Exception:
            logger.exception("Error closing DB.")