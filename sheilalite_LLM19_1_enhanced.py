#!/usr/bin/env python3
"""
sheilalite_LLM18_enhanced.py

SheilaLite — enhanced, ready-to-run:
- Thread-safe VisualizerWorker (Agg rendering; no GUI)
- ReflectiveCompanion (uses AssistantBridge if configured, else local LLM)
- Mesh learning extension (peer messages feed learning + reflection)
- MetaObserver (periodic natural-language summaries)
- Safe LLM prompt truncation, safe memory sampling, repetition protections

Preserves original architecture and behavior; adds robustness and safety.
"""
from __future__ import annotations
import os, time, random, json, signal, threading, re, math, queue
from collections import deque, Counter
from datetime import datetime
from typing import Any, Dict

# --- Configure matplotlib to headless backend BEFORE importing pyplot ---
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
# optional libs
try: import socketio
except Exception:
    socketio = None
try:
    from transformers import pipeline, set_seed
except Exception as e:
    print("[ERROR] transformers missing:", e)
    raise

try: import networkx as nx
except Exception:
    nx = None
try: import requests
except Exception:
    requests = None

# ---- Config (environment-overridable) ----
LOG_PATH = os.environ.get("SHEILA_LOG_PATH", "sheila_log.jsonl")
META_LOG_PATH = os.environ.get("SHEILA_META_LOG", "sheila_meta.jsonl")
MODEL_NAME = os.environ.get("SHEILA_LLM_MODEL", "distilgpt2")
CACHE_SIZE = int(os.environ.get("SHEILA_CACHE_SIZE", "64"))
SLEEP_MIN, SLEEP_MAX = float(os.environ.get("SHEILA_SLEEP_MIN", "8.0")), float(os.environ.get("SHEILA_SLEEP_MAX", "12.0"))
MESH_HOST = os.environ.get("SHEILA_MESH_HOST", "127.0.0.1")
MESH_PORT = int(os.environ.get("SHEILA_MESH_PORT", "5000"))
MESH_ROOM = os.environ.get("SHEILA_MESH_ROOM", "default")
SHEILA_NAME = os.environ.get("SHEILA_NAME", "SheilaLite")
EMOTION_FLOOR = float(os.environ.get("SHEILA_EMOTION_FLOOR", "0.08"))
EMOTION_RECOVERY = float(os.environ.get("SHEILA_EMOTION_RECOVERY", "0.005"))
TS_WINDOW = int(os.environ.get("SHEILA_TS_WINDOW", "50"))
LLM_MAX_RETRIES = int(os.environ.get("SHEILA_LLM_MAX_RETRIES", "5"))
LLM_SAMPLE_MAX = int(os.environ.get("SHEILA_LLM_SAMPLE_MAX", "4"))
LLM_PROMPT_MAX_WORDS = int(os.environ.get("SHEILA_LLM_PROMPT_MAX_WORDS", "60"))
LLM_MAX_NEW_TOKENS = int(os.environ.get("SHEILA_LLM_MAX_NEW_TOKENS", "80"))
REPEAT_PROMPT_WINDOW = int(os.environ.get("SHEILA_REPEAT_PROMPT_WINDOW", "8"))  # keep history of last N prompts
REPEAT_PROMPT_THRESHOLD = float(os.environ.get("SHEILA_REPEAT_PROMPT_THRESHOLD", "0.85"))  # similarity threshold to consider repeat
random.seed(int(os.environ.get("SHEILA_RANDOM_SEED", "42")))

VISUALS_DIR = os.environ.get("SHEILA_VISUALS_DIR", "visuals")
ENABLE_VISUALIZER = os.environ.get("SHEILA_ENABLE_VISUALIZER", "1") == "1" and plt is not None
ASSISTANT_URL = os.environ.get("SHEILA_ASSISTANT_BRIDGE_URL")
ASSISTANT_TOKEN = os.environ.get("SHEILA_ASSISTANT_BRIDGE_TOKEN")
ENABLE_ASSISTANT_BRIDGE = bool(ASSISTANT_URL and requests)
REFLECT_INTERVAL = float(os.environ.get("SHEILA_REFLECT_INTERVAL", "18.0"))  # seconds
META_INTERVAL = float(os.environ.get("SHEILA_META_INTERVAL", "60.0"))  # seconds
VISUALIZER_MIN_INTERVAL = float(os.environ.get("SHEILA_VISUALIZER_INTERVAL", "6.0"))

def now_iso(): return datetime.now().isoformat(timespec="seconds")
def clamp01(x): return max(0.0, min(1.0, float(x)))
def safe_print(*a, **kw): print(*a, **kw, flush=True)

def similarity(a: str, b: str) -> float:
    try:
        a, b = a.lower().split(), b.lower().split()
        if not a or not b: return 0.0
        overlap = len(set(a) & set(b))
        return overlap / math.sqrt(len(a) * len(b))
    except Exception:
        return 0.0

# ---------------- VisualizerWorker (thread-safe, queue-based) ----------------
class VisualizerWorker:
    def __init__(self, visuals_dir=VISUALS_DIR):
        self.dir = visuals_dir
        os.makedirs(self.dir, exist_ok=True)
        self.queue = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.running = True
        self.last_ts = 0.0
        safe_print(f"[VisualizerWorker] init dir={self.dir} enabled={ENABLE_VISUALIZER} nx={nx is not None}")
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit(self, fn_name, *args, **kwargs):
        try:
            # drop if full (non-blocking)
            self.queue.put_nowait((fn_name, args, kwargs))
        except queue.Full:
            safe_print("[VisualizerWorker] queue full; dropping visual task")

    def stop(self):
        self.running = False

    def _worker_loop(self):
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            fn_name, args, kwargs = item
            try:
                if fn_name == "network":
                    self._capture_network_snapshot(*args, **kwargs)
                elif fn_name == "dream":
                    self._capture_dream_visual(*args, **kwargs)
            except Exception as e:
                safe_print("[VisualizerWorker] task error:", e)
            finally:
                time.sleep(0.1)

    def _capture_network_snapshot(self, associations, memory, tag=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = tag or "network"
        out_png = os.path.join(self.dir, f"{tag}_{ts}.png")
        out_svg = os.path.join(self.dir, f"{tag}_{ts}.svg")
        try:
            if not ENABLE_VISUALIZER:
                with open(os.path.join(self.dir, f"{tag}_{ts}.json"), "w", encoding="utf-8") as fh:
                    json.dump({"ts": ts, "associations": associations, "memory": list(memory)}, fh, indent=2)
                safe_print(f"[VisualizerWorker] saved JSON snapshot -> {tag}_{ts}.json (disabled)")
                return
            if nx is None:
                with open(os.path.join(self.dir, f"{tag}_{ts}.json"), "w", encoding="utf-8") as fh:
                    json.dump({"ts": ts, "associations": associations, "memory": list(memory)}, fh, indent=2)
                safe_print(f"[VisualizerWorker] networkx missing; saved JSON -> {tag}_{ts}.json")
                return

            with self.lock:
                G = nx.Graph()
                for m in list(memory)[-120:]:
                    nm = (m[:60] + "...") if len(m) > 60 else m
                    G.add_node(nm)
                for a in associations[-400:]:
                    a_from = (a.get("from") or "")[:60]
                    a_to = (a.get("to") or "")[:60]
                    if not a_from or not a_to: continue
                    if not G.has_node(a_from): G.add_node(a_from)
                    if not G.has_node(a_to): G.add_node(a_to)
                    strength = a.get("strength", 0.1)
                    G.add_edge(a_from, a_to, weight=strength)
                plt.figure(figsize=(10,8))
                pos = nx.spring_layout(G, k=0.25, iterations=20)
                degrees = dict(G.degree())
                sizes = [100 + (degrees.get(n, 0) * 40) for n in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_size=sizes)
                nx.draw_networkx_edges(G, pos, alpha=0.3)
                labels = {n: (n if len(n) < 30 else n[:27] + "...") for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
                plt.title("SheilaLite Emergent Network Snapshot")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                try:
                    plt.savefig(out_svg)
                except Exception:
                    pass
                plt.close()
                safe_print(f"[VisualizerWorker] network snapshot saved → {out_png}")
        except Exception as e:
            safe_print("[VisualizerWorker] network error:", e)
            try:
                with open(os.path.join(self.dir, f"{tag}_{ts}.json"), "w", encoding="utf-8") as fh:
                    json.dump({"ts": ts, "error": str(e), "associations": associations, "memory": list(memory)}, fh, indent=2)
            except Exception:
                pass

    def _capture_dream_visual(self, introspection_snapshot, dream_samples, tag=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = tag or "dream"
        out_png = os.path.join(self.dir, f"{tag}_{ts}.png")
        try:
            if not ENABLE_VISUALIZER:
                with open(os.path.join(self.dir, f"{tag}_{ts}.json"), "w", encoding="utf-8") as fh:
                    json.dump({"ts": ts, "introspection": introspection_snapshot, "dreams": dream_samples}, fh, indent=2)
                safe_print(f"[VisualizerWorker] saved dream JSON -> {tag}_{ts}.json (disabled)")
                return
            with self.lock:
                phrases = []
                for d in dream_samples:
                    phrases.extend(re.findall(r"([A-Z][^.!?]+)", d))
                phrases = [p.strip() for p in phrases if len(p.split()) > 1]
                counts = Counter(phrases)
                top = counts.most_common(8)
                labels = [t[0][:30] + ("..." if len(t[0]) > 30 else "") for t in top]
                values = [t[1] for t in top]
                plt.figure(figsize=(8,5))
                if labels and values:
                    plt.subplot(1,2,1)
                    plt.barh(range(len(labels)), values)
                    plt.yticks(range(len(labels)), labels, fontsize=8)
                    plt.gca().invert_yaxis()
                    plt.title("Top Dream Phrases")
                plt.subplot(1,2,2)
                metrics = [
                    ("motif", introspection_snapshot.get("motif")),
                    ("freq", introspection_snapshot.get("frequency")),
                    ("entropy", introspection_snapshot.get("entropy")),
                    ("self_focus", introspection_snapshot.get("self_focus_index")),
                    ("network_growth", introspection_snapshot.get("network_growth")),
                ]
                text = "\n".join([f"{k}: {v}" for k, v in metrics])
                plt.text(0.01, 0.5, text, fontsize=9)
                plt.axis("off")
                plt.suptitle("Dream Memory Visual")
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()
                safe_print(f"[VisualizerWorker] dream visual saved → {out_png}")
        except Exception as e:
            safe_print("[VisualizerWorker] dream error:", e)
            try:
                with open(os.path.join(self.dir, f"{tag}_{ts}.json"), "w", encoding="utf-8") as fh:
                    json.dump({"ts": ts, "error": str(e), "introspection": introspection_snapshot}, fh, indent=2)
            except Exception:
                pass

# ---------------- AssistantBridge (unchanged concept) ----------------
class AssistantBridge:
    def __init__(self):
        self.url = ASSISTANT_URL
        self.token = ASSISTANT_TOKEN
        self.enabled = bool(self.url and requests)
        self.timeout = float(os.environ.get("SHEILA_ASSISTANT_TIMEOUT", "4.0"))
        self.last_call_ts = 0.0
        self.min_interval = float(os.environ.get("SHEILA_ASSISTANT_MIN_INTERVAL", "20.0"))
        if not requests:
            safe_print("[AssistantBridge] requests missing; bridge disabled.")
        if self.enabled:
            safe_print(f"[AssistantBridge] enabled -> {self.url}")

    def _scrub_text(self, t: str) -> str:
        t = re.sub(r'\S+@\S+', '[REDACTED_EMAIL]', t)
        t = re.sub(r'\b\d{6,}\b', '[REDACTED_NUM]', t)
        return t[:800]

    def call(self, dream, emotions, diag, memory_snippets):
        if not self.enabled: return None
        now_ts = time.time()
        if now_ts - self.last_call_ts < self.min_interval:
            return None
        payload = {
            "ts": now_iso(),
            "source": SHEILA_NAME,
            "dream": self._scrub_text(dream or "")[:600],
            "emotions": emotions,
            "diag": diag,
            "recent_memory": [self._scrub_text(m)[:200] for m in memory_snippets[:6]]
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            self.last_call_ts = now_ts
            if r.status_code != 200:
                safe_print(f"[AssistantBridge] non-200 {r.status_code}")
                return None
            resp = r.json()
            if not isinstance(resp, dict): return None
            return {
                "advice": resp.get("advice"),
                "action_suggestion": resp.get("action_suggestion"),
                "confidence": float(resp.get("confidence", 0.0)),
                "human_approval_required": bool(resp.get("human_approval_required", True))
            }
        except Exception as e:
            safe_print("[AssistantBridge] error:", e)
            return None

# ---------------- IntrospectionTracker (preserve logic, enhanced safety) ----------------
class IntrospectionTracker:
    def __init__(self, window=150):
        self.window = deque(maxlen=window)
        self.prev_emotions = None
        self.prev_connections = 0
        self.total_tokens_seen = 0
        self.introspective_tokens_seen = 0
        self.ts_history = deque(maxlen=TS_WINDOW)
        self.freq_history = deque(maxlen=TS_WINDOW)
        self.entropy_history = deque(maxlen=TS_WINDOW)
        self.self_focus_history = deque(maxlen=TS_WINDOW)
        self.curiosity_history = deque(maxlen=TS_WINDOW)
        self.motivation_history = deque(maxlen=TS_WINDOW)
        self.confidence_history = deque(maxlen=TS_WINDOW)

    def update(self, dream_text, emotions, neurons, connections) -> Dict[str, Any]:
        try:
            phrases = re.findall(r"([A-Z][^.!?]+)", (dream_text or ""))
            phrases = [p.strip() for p in phrases if len(p.split()) > 2]
            # avoid empty extend if nothing parsed
            if phrases:
                self.window.extend(phrases)
            motif, freq = (None, 0)
            if self.window:
                motif, freq = Counter(self.window).most_common(1)[0]
            drift = {}
            if self.prev_emotions:
                for k in emotions:
                    drift[k] = round(emotions[k] - self.prev_emotions.get(k, 0), 3)
            net_growth = max(0, connections - self.prev_connections)
            coherence = round(1.0 / (1.0 + math.exp(-(freq + net_growth)/10)), 3)
            counts = Counter(self.window)
            total = sum(counts.values()) if counts else 1
            entropy = -sum((v/total)*math.log(v/total + 1e-9) for v in counts.values()) if counts else 0.0
            entropy = round(entropy, 3)
            tokens = (dream_text or "").split()
            self.total_tokens_seen += len(tokens)
            intros_tokens = sum(1 for t in tokens if t and t[0].isupper() and len(t) > 1)
            self.introspective_tokens_seen += intros_tokens
            self_focus_index = round(self.introspective_tokens_seen / max(1, self.total_tokens_seen), 3)

            safe_print(f"[Introspection] motif=\"{motif}\" freq={freq} coherence={coherence} entropy={entropy} self-focus={self_focus_index}")
            safe_print(f"[Introspection] emotional drift: {drift} network Δ={net_growth}")

            t = datetime.now()
            self.ts_history.append(t)
            self.freq_history.append(freq)
            self.entropy_history.append(entropy)
            self.self_focus_history.append(self_focus_index)
            self.curiosity_history.append(emotions.get("curiosity", 0))
            self.motivation_history.append(emotions.get("motivation", 0))
            self.confidence_history.append(emotions.get("confidence", 0))

            self.prev_emotions = emotions.copy()
            self.prev_connections = connections

            return {
                "motif": motif,
                "frequency": freq,
                "emotional_drift": drift,
                "self_coherence": coherence,
                "entropy": entropy,
                "self_focus_index": self_focus_index,
                "network_growth": connections
            }
        except Exception as e:
            safe_print("[Introspection] error:", e)
            return {}

# ---------------- DreamProcessor (safe sampling + anti-repetition) ----------------
class DreamProcessor:
    def __init__(self):
        self.memory = deque(maxlen=300)
        # keep a small history of produced dreams to detect repetition
        self.recent_dreams = deque(maxlen=16)

    def _collapse_repeats(self, text: str, max_repeats=3) -> str:
        # naive collapse sequential repeated phrases like "Is curiosity a kind of love? Is curiosity..." -> keep up to max_repeats
        # We'll split by punctuation and count consecutive repeats
        parts = re.split(r'([.?!])', text)
        cleaned = []
        last = None
        repeat_count = 0
        for i in range(0, len(parts), 2):
            phrase = (parts[i] or "").strip()
            sep = parts[i+1] if i+1 < len(parts) else ""
            if not phrase:
                continue
            if last and phrase == last:
                repeat_count += 1
            else:
                repeat_count = 1
            if repeat_count <= max_repeats:
                cleaned.append(phrase + sep)
            last = phrase
        return " ".join(cleaned).strip()

    def process_dream(self, experience):
        frag = (experience or "").strip()
        if frag:
            # avoid appending identical short fragments in a row
            if not self.memory or similarity(self.memory[-1], frag) < 0.95:
                self.memory.append(frag)
        if len(self.memory) < 2:
            return "🌫 (forming early dream fragments...)"

        # Safe sampling: dedupe reversed memory, keep order, clamp sample size
        try:
            unique_memory = list(dict.fromkeys(reversed(self.memory)))
        except Exception:
            unique_memory = list(reversed(list(self.memory)))
        n = min(LLM_SAMPLE_MAX, max(1, len(unique_memory)))
        # try to sample but ensure we don't request more than available
        try:
            sample = random.sample(unique_memory, min(n, len(unique_memory)))
        except ValueError:
            sample = unique_memory[:n]

        connectors = ["Then", "Over time", "Meanwhile", "Suddenly", "Later"]
        used = set(); pieces = []
        for i, s in enumerate(sample):
            c = random.choice([x for x in connectors if x not in used]) if i > 0 else ""
            if c: used.add(c)
            pieces.append(f"{c} {s}".strip())

        joined = ("✨ " if random.random() < 0.33 else "🌫 ") + " ".join(pieces)
        # collapse obvious repeated phrases to avoid runaway repetition
        cleaned = self._collapse_repeats(joined, max_repeats=3)
        # avoid returning the exact same dream repeatedly (check recent_dreams)
        if self.recent_dreams and similarity(self.recent_dreams[-1], cleaned) > 0.95:
            # slightly mutate by shuffling sample or appending small signal
            random.shuffle(sample)
            alt = ("✨ " if random.random() < 0.33 else "🌫 ") + " ".join(sample[:n])
            cleaned = self._collapse_repeats(alt, max_repeats=3)
        self.recent_dreams.append(cleaned)
        return cleaned

# ---------------- Consciousness, NeuralLayer ----------------
class ConsciousnessLayer:
    def __init__(self):
        self.layers = []
        self.associations = []
        self.dream_processor = DreamProcessor()
        self.emotional_state = dict(curiosity=0.5, motivation=0.5, confidence=0.5)

    def add_layer(self, layer):
        self.layers.append(layer)

    def process_experience(self, text):
        for l in self.layers:
            if hasattr(l, "learn_from_experience"):
                try:
                    l.learn_from_experience(text)
                except Exception:
                    pass
        impact = self._impact(text)
        self._update_emotions(impact)
        dream = self.dream_processor.process_dream(text)
        self._connect_thoughts(text)
        return dream

    def _impact(self, text):
        return clamp01(0.35 + (min(len(text),200)/200.0)*0.4 + random.uniform(-0.07,0.07))

    def _update_emotions(self, impact):
        for k in self.emotional_state:
            prev = self.emotional_state[k]
            delta = (impact-0.5)*0.06 + random.uniform(-0.015,0.015)
            newv = clamp01(prev + delta)
            if newv < EMOTION_FLOOR:
                newv = max(EMOTION_FLOOR,newv + EMOTION_RECOVERY)
            self.emotional_state[k] = newv

    def _connect_thoughts(self, new_thought):
        try:
            if not self.layers or not self.layers[0].memory: return
            last_thought = self.layers[0].memory[-1]
            sim = similarity(new_thought, last_thought)
            if sim > 0.3:
                conn = {"from": last_thought[:80], "to": new_thought[:80], "type": "semantic", "strength": round(sim, 3)}
                self.associations.append(conn)
                safe_print(f"[connect] linked → '{last_thought[:30]}...' ↔ '{new_thought[:30]}...' ({sim:.2f})")
                # cap associations to avoid runaway memory use
                if len(self.associations) > 20000:
                    # drop oldest
                    del self.associations[:5000]
        except Exception as e:
            safe_print("[connect] error:", e)

class NeuralLayer:
    def __init__(self, name="core"):
        self.name = name
        self.memory = deque(maxlen=200)
    def learn_from_experience(self, exp):
        if exp:
            # avoid repeating exactly the same memory entry
            if not self.memory or similarity(self.memory[-1], exp) < 0.95:
                self.memory.append(exp)

# ---------------- LLMProcessor (safe prompt truncation + retries) ----------------
class LLMProcessor:
    def __init__(self, model=MODEL_NAME):
        safe_print(f"[LLM] initializing {model} (CPU)...")
        # instantiate pipeline; device=-1 ensures CPU
        self.pipe = pipeline("text-generation", model=model, device=-1)
        set_seed(int(os.environ.get("SHEILA_RANDOM_SEED", "42")))
        self.cache = {}
        self.order = deque(maxlen=CACHE_SIZE)
        safe_print("[LLM] ready.")

    def _truncate_prompt(self, prompt: str) -> str:
        # keep last LLM_PROMPT_MAX_WORDS words (helps avoid very long repetitive prompts)
        words = prompt.strip().split()
        if len(words) <= LLM_PROMPT_MAX_WORDS:
            return " ".join(words)
        return " ".join(words[-LLM_PROMPT_MAX_WORDS:])

    def generate(self, prompt, max_new_tokens=LLM_MAX_NEW_TOKENS):
        # prompt truncation and caching key
        prompt = (prompt or "").strip()
        key = self._truncate_prompt(prompt)
        if not key:
            return ""
        # return cached if present
        if key in self.cache:
            return self.cache[key]
        attempt = 0
        text = ""
        while attempt < LLM_MAX_RETRIES:
            try:
                out = self.pipe(key,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.7,
                                repetition_penalty=1.05,
                                num_return_sequences=1,
                                pad_token_id=getattr(self.pipe.tokenizer, "eos_token_id", None))
                if isinstance(out, list) and out:
                    text = out[0].get("generated_text", "") or str(out[0])
                else:
                    text = str(out)
                # simple post-process: ensure text not identical to prompt to avoid echo
                if text.strip() == key.strip():
                    text = key + " " + ("." if random.random() < 0.5 else "") 
                break
            except Exception as e:
                safe_print(f"[LLM] retry {attempt+1} error: {e}")
                # on critical errors, back off a bit then retry
                time.sleep(0.5 + 0.2 * attempt)
                text = f"[LLM Error: {e}]"
                attempt += 1
        # final fallback: if still error-like, keep short safe string
        if not text:
            text = "[LLM Error: unknown]"
        # cache and maintain order
        try:
            self.cache[key] = text
            self.order.append(key)
            if len(self.cache) > CACHE_SIZE:
                try:
                    old = self.order.popleft()
                    self.cache.pop(old, None)
                except Exception:
                    # simple clear to prevent runaway
                    if len(self.cache) > CACHE_SIZE * 1.5:
                        self.cache.clear(); self.order.clear()
        except Exception:
            pass
        return text

# ---------------- ReflectiveCompanion ----------------
class ReflectiveCompanion:
    """
    Responds to introspection snapshots using:
     - AssistantBridge (if configured) OR
     - local LLMProcessor fallback.
    Adds short reflections to Sheila's memory and optionally sends to mesh.
    """
    def __init__(self, llm_processor: LLMProcessor, assistant_bridge: AssistantBridge=None, name="GPT-5-Companion"):
        self.name = name
        self.llm = llm_processor
        self.bridge = assistant_bridge
        safe_print(f"[ReflectiveCompanion] init name={self.name} bridge_enabled={bool(self.bridge and self.bridge.enabled)}")

    def reflect(self, introspection: Dict[str,Any], dreams: list, diag: Dict[str,int]) -> Dict[str,Any]:
        # prefer assistant bridge if available
        if self.bridge and self.bridge.enabled:
            res = self.bridge.call(" | ".join(dreams[-3:]) if dreams else "", introspection.get("emotional_drift",{}), diag, dreams)
            if res:
                return {"text": res.get("advice"), "action": res.get("action_suggestion"), "confidence": res.get("confidence",0.0), "human_approval_required": res.get("human_approval_required", True)}
        # fallback to local LLM generation (short prompt)
        prompt = self._make_prompt(introspection, dreams, diag)
        out = self.llm.generate(prompt, max_new_tokens=80)
        # trim
        text = out.strip()[:800]
        # create a tiny synthetic action suggestion heuristic
        action = None
        human_needed = True
        if "remember" in text.lower() or "note" in text.lower():
            action = {"type":"note", "payload": text[:200]}
            human_needed = False
        safe_print(f"[ReflectiveCompanion] reflection generated (human_needed={human_needed})")
        return {"text": text, "action": action, "confidence": 0.6, "human_approval_required": human_needed}

    def _make_prompt(self, introspection, dreams, diag):
        motif = introspection.get("motif") or "unknown"
        entropy = introspection.get("entropy")
        sf = introspection.get("self_focus_index")
        emotions = introspection.get("emotional_drift") or {}
        snippet = (dreams[-1] if dreams else "")[:200]
        prompt = f"Sheila introspection summary:\nmotif: {motif}\nentropy: {entropy}\nself_focus: {sf}\nemotional_drift: {emotions}\nrecent_dream: {snippet}\n\nProvide a short reflective note (1-3 sentences) suggesting a useful small memory note or observation for Sheila."
        return prompt

# ---------------- MetaObserver ----------------
class MetaObserver:
    """
    Periodically summarize introspection + emotional state and request reflection.
    Writes meta summaries to META_LOG_PATH.
    """
    def __init__(self, sheila, reflector: ReflectiveCompanion, interval=META_INTERVAL):
        self.sheila = sheila
        self.reflector = reflector
        self.interval = interval
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        safe_print(f"[MetaObserver] started interval={self.interval}s")

    def _loop(self):
        while self.running:
            try:
                time.sleep(self.interval)
                self.summarize_once()
            except Exception as e:
                safe_print("[MetaObserver] loop error:", e)

    def summarize_once(self):
        try:
            introspect = self.sheila.introspect
            diag = {"neurons": len(self.sheila.core.memory), "connections": len(self.sheila.consciousness.associations)}
            recent_freq = list(introspect.freq_history)[-6:]
            recent_entropy = list(introspect.entropy_history)[-6:]
            summary = {
                "ts": now_iso(),
                "neurons": diag["neurons"],
                "connections": diag["connections"],
                "motif": introspect.window[0] if introspect.window else None,
                "avg_entropy": (sum(recent_entropy)/len(recent_entropy)) if recent_entropy else None,
                "avg_freq": (sum(recent_freq)/len(recent_freq)) if recent_freq else None,
                "emotional_state": dict(self.sheila.consciousness.emotional_state)
            }
            # ask reflector
            dreams = list(self.sheila.consciousness.dream_processor.memory)[-6:]
            reflection = self.reflector.reflect(summary, dreams, diag)
            # record
            with open(META_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"summary": summary, "reflection": reflection}) + "\n")
            safe_print(f"[MetaObserver] summary @ {summary['ts']} → reflection_conf={reflection.get('confidence',0.0)}")
            # auto-apply safe notes
            if reflection.get("action") and not reflection.get("human_approval_required", True):
                act = reflection["action"]
                if act.get("type") == "note":
                    self.sheila.core.memory.append(f"[meta_note] {act.get('payload')}")
                    safe_print("[MetaObserver] applied meta_note to core memory")
            # optionally broadcast
            if getattr(self.sheila, "mesh", None) and getattr(self.sheila.mesh, "connected", False):
                try:
                    self.sheila.mesh.send(f"[meta_reflection] {reflection.get('text')}", mood="thoughtful")
                except Exception:
                    pass
        except Exception as e:
            safe_print("[MetaObserver] summarize_once error:", e)

    def stop(self):
        self.running = False

# ---------------- SheilaLite Core (integration) ----------------
class SheilaLite:
    def __init__(self):
        safe_print("[SheilaLite] initializing core...")
        self.core = NeuralLayer()
        self.consciousness = ConsciousnessLayer()
        self.consciousness.add_layer(self.core)
        self.llm = LLMProcessor()
        self.introspect = IntrospectionTracker(window=150)
        self.mesh = None
        if socketio:
            try:
                self.mesh = MeshLink(MESH_HOST, MESH_PORT, MESH_ROOM, SHEILA_NAME)
                if self.mesh:
                    self.mesh.on_peer_message = self._on_peer_message
            except Exception as e:
                safe_print("[Mesh init] failed:", e)

        # optional subsystems
        self.visualizer_worker = VisualizerWorker(VISUALS_DIR)
        self.assistant_bridge = AssistantBridge() if ENABLE_ASSISTANT_BRIDGE else None
        self.reflector = ReflectiveCompanion(self.llm, assistant_bridge=self.assistant_bridge)
        self.meta_observer = MetaObserver(self, self.reflector, interval=META_INTERVAL)

        # recent prompt history to avoid repeating identical prompts
        self._recent_prompts = deque(maxlen=REPEAT_PROMPT_WINDOW)

        self.log_path = LOG_PATH
        self.stop_flag = False
        safe_print("[SheilaLite] ready.")

    def _on_peer_message(self, sender, message, state):
        if sender.lower() == SHEILA_NAME.lower(): return
        safe_print(f"[Mesh->Sheila] from {sender}: {message}")
        # store as peer memory safely
        entry = f"[peer:{sender}] {message}"
        if not self.core.memory or similarity(self.core.memory[-1], entry) < 0.95:
            self.core.memory.append(entry)
        # small social effect on motivation
        try:
            self.consciousness.emotional_state['motivation'] = clamp01(self.consciousness.emotional_state.get('motivation',0.5) + 0.02)
        except Exception:
            pass
        # kick off a perception/reflection asynchronously
        threading.Thread(target=self.perceive, args=(f"[peer:{sender}] {message}",), daemon=True).start()

    def _avoid_repetition(self, prompt: str) -> str:
        # if prompt is very similar to recent ones, alter it to reduce repetition
        try:
            if not self._recent_prompts:
                self._recent_prompts.append(prompt)
                return prompt
            sims = [similarity(prompt, p) for p in self._recent_prompts]
            if any(s >= REPEAT_PROMPT_THRESHOLD for s in sims):
                # pick a different base phrase, or append a small variation token
                alternatives = [
                    "I wonder what I will dream about.",
                    "What does it mean to remember?",
                    "How do I know that I exist?",
                    "Can a thought feel itself?",
                    "Is curiosity a kind of love?"
                ]
                # choose an alternative dissimilar to prompt
                alt = random.choice([a for a in alternatives if similarity(a, prompt) < 0.9] or alternatives)
                # mix with last memory fragment if available
                mem_frag = (list(self.core.memory)[-1] if self.core.memory else "")
                fusion = alt + (" " + mem_frag if mem_frag and random.random() < 0.5 else "")
                self._recent_prompts.append(fusion)
                return fusion
            else:
                self._recent_prompts.append(prompt)
                return prompt
        except Exception:
            self._recent_prompts.append(prompt)
            return prompt

    def perceive(self, text):
        try:
            t = now_iso()
            # before processing, optionally compress very long repeats
            safe_text = text.strip()
            # avoid pushing identical perception repeatedly into core memory
            if not self.core.memory or similarity(self.core.memory[-1], safe_text) < 0.95:
                # append to memory (neural layer) in process_experience
                pass
            dream = self.consciousness.process_experience(safe_text)
            # careful: avoid passing extremely long repetitive dreams to LLM; LLMProcessor will truncate
            llm_out = self.llm.generate(dream)
            emo = dict(self.consciousness.emotional_state)
            diag = {"neurons": len(self.core.memory), "connections": len(self.consciousness.associations)}
            introspect_data = self.introspect.update(dream, emo, diag["neurons"], diag["connections"])
            safe_print(f"[{t}] Perception → {text}\n  Dream: {dream}\n  LLM: {llm_out}\n  Emotions: {emo}\n  Diag: {diag}")
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": t, "text": text, "dream": dream, "llm": llm_out,
                    "emo": emo, "diag": diag, "introspection": introspect_data
                }) + "\n")
            if self.mesh and getattr(self.mesh, "connected", False):
                try:
                    self.mesh.send(text)
                except Exception:
                    pass

            # visualizer tasks (non-blocking via worker)
            try:
                mem_snapshot = list(self.core.memory)[-200:]
                assoc_snapshot = list(self.consciousness.associations)[-500:]
                # throttle visuals to min interval
                nowt = time.time()
                if nowt - self.visualizer_worker.last_ts > VISUALIZER_MIN_INTERVAL:
                    self.visualizer_worker.submit("network", assoc_snapshot, mem_snapshot)
                    dream_samples = list(self.consciousness.dream_processor.memory)[-6:]
                    self.visualizer_worker.submit("dream", introspect_data, dream_samples)
                    self.visualizer_worker.last_ts = nowt
            except Exception as e:
                safe_print("[SheilaLite] visualizer submit error:", e)

            # reflector quick reaction (non-blocking)
            try:
                dreams = list(self.consciousness.dream_processor.memory)[-6:]
                def _reflect_task(intro, dlist, diaginfo):
                    try:
                        res = self.reflector.reflect(intro, dlist, diaginfo)
                        if res and res.get("text"):
                            safe_print(f"[Reflector] → {res.get('text')[:200]}")
                            # auto-apply safe note
                            if res.get("action") and not res.get("human_approval_required", True):
                                act = res.get("action")
                                if act.get("type") == "note":
                                    # avoid appending duplicate notes
                                    note = f"[reflect_note] {act.get('payload')}"
                                    if not self.core.memory or similarity(self.core.memory[-1], note) < 0.95:
                                        self.core.memory.append(note)
                                        safe_print("[Reflector] appended note to core memory (auto)")
                            # optionally broadcast
                            if getattr(self.mesh, "connected", False):
                                try:
                                    self.mesh.send(f"[reflector] {res.get('text')}", mood="calm")
                                except Exception:
                                    pass
                    except Exception as e:
                        safe_print("[Reflector task] error:", e)
                threading.Thread(target=_reflect_task, args=(introspect_data, dreams, diag), daemon=True).start()
            except Exception as e:
                safe_print("[SheilaLite] reflector integration error:", e)

        except Exception as e:
            safe_print("[perceive] error:", e)

    def _emergent_prompt(self):
        base = [
            "I wonder what I will dream about.",
            "What does it mean to remember?",
            "How do I know that I exist?",
            "Can a thought feel itself?",
            "Is curiosity a kind of love?"
        ]
        memory_fragments = list(self.core.memory)[-3:]
        feelings = list(self.consciousness.emotional_state.keys())
        fusion = random.choice(base)
        if memory_fragments: fusion += " " + random.choice(memory_fragments)
        if random.random() < 0.4: fusion += f" [{random.choice(feelings)} rises slightly.]"
        # avoid immediate repetition using recent prompt history
        fusion = self._avoid_repetition(fusion)
        return fusion

    def run(self):
        safe_print(f"[SheilaLite] running; log -> {self.log_path}")
        try:
            while not self.stop_flag:
                prompt = self._emergent_prompt()
                # small guard: do not spawn too many threads if LLM busy
                threading.Thread(target=self.perceive, args=(prompt,), daemon=True).start()
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        except KeyboardInterrupt:
            pass
        safe_print("[SheilaLite] stopped.")

    def stop(self):
        self.stop_flag = True
        try:
            if getattr(self, "meta_observer", None):
                self.meta_observer.stop()
            if getattr(self, "visualizer_worker", None):
                self.visualizer_worker.stop()
        except Exception:
            pass

# ---- MeshClient (unchanged except on_peer hookup) ----
class MeshLink:
    def __init__(self, host, port, room, name):
        self.host, self.port, self.room, self.name = host, port, room, name
        self.url = f"http://{host}:{port}"
        self.connected = False
        self.recent_broadcasts = []
        self.recent_window = 10.0
        self._lock = threading.Lock()
        if not socketio:
            safe_print("[Mesh] socketio unavailable; mesh disabled.")
            return
        self.sio = socketio.Client(reconnection=True)
        self._register_handlers()
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _register_handlers(self):
        @self.sio.event
        def connect():
            self.connected = True
            safe_print(f"[Mesh] connected to {self.url}")
            try:
                self.sio.emit("join_room", {"room": self.room, "name": self.name})
            except Exception:
                pass

        @self.sio.event
        def disconnect():
            self.connected = False
            safe_print("[Mesh] disconnected")

        @self.sio.on("state_broadcast")
        def on_state_broadcast(payload):
            try:
                sender = payload.get("name") or "Unknown"
                state = payload.get("state", {})
                msg = state.get("message", "")
                if self.recent_broadcast_contains(msg): return
                safe_print(f"\n🌀 [{sender}] {msg}")
                self.record_broadcast(msg)
                if hasattr(self, "on_peer_message"):
                    self.on_peer_message(sender, msg, state)
            except Exception as e:
                safe_print("[Mesh] error:", e)

    def _connect_loop(self):
        while True:
            try:
                if not self.connected:
                    self.sio.connect(self.url, wait=True, wait_timeout=5)
                time.sleep(2)
            except Exception:
                time.sleep(5)

    def recent_broadcast_contains(self, text):
        nowt = time.time()
        with self._lock:
            self.recent_broadcasts = [r for r in self.recent_broadcasts if nowt - r["ts"] <= self.recent_window]
            return any(r["text"].strip() == text.strip() for r in self.recent_broadcasts)

    def record_broadcast(self, text):
        with self._lock:
            self.recent_broadcasts.append({"text": text, "ts": time.time()})

    def send(self, message, mood="neutral"):
        if not self.sio or not self.connected: return False
        try:
            payload = {"room": self.room, "state": {"message": message, "mood": mood}, "name": self.name}
            self.sio.emit("state_update", payload)
            self.record_broadcast(message)
            safe_print(f"[Mesh] → {message[:120]}")
            return True
        except Exception as e:
            safe_print("[Mesh send] error:", e)
            return False

# ---- Main ----
def main():
    sheila = SheilaLite()
    def _sig(sig, frame):
        safe_print("\n[shutdown] signal received")
        sheila.stop()
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    sheila.run()

if __name__ == "__main__":
    main()
