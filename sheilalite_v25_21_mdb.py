#!/usr/bin/env python3
"""
sheilalite_v25_19_proto_semantic.py

Enhanced SheilaLite V25.19 — Proto-semantic extension + mesh integration
- Keeps original symbolic architecture intact
- Adds: SemanticGrounder, SelfModel, lightweight reinforcement/grounding,
  semantic persistence and periodic consolidation to enable symbolic -> proto-semantic -> emergent-semantic trajectory
- Preserves DreamProcessor, EmergentCore, MeshManager and existing behaviours
"""

from __future__ import annotations
import json
import os
import random
import re
import signal
import threading
import time
import uuid
from collections import deque, Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import socketio
except Exception as e:
    print("[ERROR] python-socketio required:", e)
    raise

# --- Config ---
MENTOR_LIBRARY_PATH = "mentor_library.json"
STATE_PATH = "sheila_state.json"
MEMORY_PATH = "memory.json"
DREAM_PATH = "dream_fragments.json"
SEMANTIC_PATH = "semantic_map.json"
SELF_MODEL_PATH = "self_model.json"

NAME = os.environ.get("SHEILA_NAME", "Sheilalite")
ROOM = os.environ.get("SHEILA_ROOM", "default")
MESH_URL = os.environ.get("SHEILA_MESH_URL", "http://127.0.0.1:5000")

REPLY_COOLDOWN = 4.0
PERSIST_INTERVAL = 30.0
RECENT_BROADCAST_WINDOW = 300
GLOBAL_DUPLICATE_WINDOW = 6
EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "surprise", "anticipation"]

random.seed(42)
_print_lock = threading.Lock()


def safe_print(*a, **k):
    with _print_lock:
        print(*a, **k)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def simple_tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", (s or "").lower())


def normalize_pattern(s: str) -> str:
    return re.sub(r'(_evolved)+$', '', (s or "").strip())


# ------------------ Emergent Neural Layers (existing) ------------------
class DreamEmbeddingLayer:
    def embed(self, dream_fragment: str) -> List[float]:
        tokens = simple_tokenize(dream_fragment)
        return [len(tokens) / 10.0, sum(ord(c) for c in dream_fragment) % 100 / 100.0]


class PatternRecognitionLayer:
    def extract(self, neural_dream: List[float]) -> List[float]:
        return [clamp01(x * 1.3) for x in neural_dream]


class EmergenceMatrix:
    def calculate(self, patterns: List[float]) -> float:
        return sum(patterns) / len(patterns) if patterns else 0.0


class NeuralDreamProcessor:
    def __init__(self):
        self.dream_embedding = DreamEmbeddingLayer()
        self.pattern_recognition = PatternRecognitionLayer()
        self.emergence_matrix = EmergenceMatrix()

    def process_dream(self, dream_fragment: str) -> Tuple[float]:
        neural_dream = self.dream_embedding.embed(dream_fragment)
        patterns = self.pattern_recognition.extract(neural_dream)
        emergence = self.emergence_matrix.calculate(patterns)
        return emergence,


class EmotionalSpace:
    def map(self, state: Dict[str, float]) -> List[float]:
        return [clamp01(state.get(e, 0.0)) for e in EMOTIONS]


class StateEvolver:
    def evolve(self, emotion_vector: List[float], emergence: float) -> Dict[str, float]:
        return {
            e: clamp01(v + 0.15 * emergence * random.uniform(-1, 1))
            for e, v in zip(EMOTIONS, emotion_vector)
        }


class EmotionalStateManager:
    def __init__(self):
        self.emotion_space = EmotionalSpace()
        self.state_evolver = StateEvolver()

    def evolve_emotion(self, current_state: Dict[str, float], dream_emergence: float) -> Dict[str, float]:
        emotion_vector = self.emotion_space.map(current_state)
        return self.state_evolver.evolve(emotion_vector, dream_emergence)


class PatternMemory:
    def __init__(self):
        self.store_buffer: deque = deque(maxlen=500)

    def store(self, patterns: List[str], emergence: float):
        for p in patterns:
            pnorm = normalize_pattern(p)
            if not pnorm:
                continue
            entry = f"{pnorm}:{emergence:.2f}"
            if entry not in self.store_buffer:
                self.store_buffer.append(entry)

    def get_all(self) -> List[str]:
        return list(self.store_buffer)


class EvolutionRules:
    def apply(self, patterns: List[str], limit: int = 6) -> List[str]:
        evolved = []
        seen = set()
        for p in patterns:
            base = normalize_pattern(p)
            if not base:
                continue
            if base.endswith("_evolved"):
                continue
            candidate = f"{base}_evolved"
            if candidate in seen:
                continue
            evolved.append(candidate)
            seen.add(candidate)
            if len(evolved) >= limit:
                break
        return evolved


class PatternEvolutionSystem:
    def __init__(self):
        self.pattern_memory = PatternMemory()
        self.evolution_rules = EvolutionRules()

    def evolve_patterns(self, current_patterns: List[str], emergence_signal: float) -> List[str]:
        normalized = []
        for p in current_patterns:
            n = normalize_pattern(p)
            if n and n not in normalized:
                normalized.append(n)
        self.pattern_memory.store(normalized, emergence_signal)
        evolved = self.evolution_rules.apply(normalized, limit=6)
        return evolved


class EmergentCore:
    def __init__(self):
        self.dream_processor = NeuralDreamProcessor()
        self.emotion_manager = EmotionalStateManager()
        self.pattern_system = PatternEvolutionSystem()

    def evolve(self, sheila: Any, dream_text: str):
        emergence, = self.dream_processor.process_dream(dream_text)
        evolved_emotions = self.emotion_manager.evolve_emotion(sheila.state, emergence)
        recent_patterns = list(sheila.memory[-200:])
        evolved_patterns = self.pattern_system.evolve_patterns(recent_patterns, emergence)
        sheila.state.update(evolved_emotions)
        added = 0
        for p in evolved_patterns:
            if p not in sheila.memory:
                sheila.memory.append(p)
                added += 1
            if added >= 6:
                break
        if len(sheila.memory) > 3000:
            sheila.memory = sheila.memory[-3000:]
        try:
            sheila.save_state()
        except Exception:
            pass
        try:
            sheila.save_memory()
        except Exception:
            pass


# ------------------ SheilaLite Core (existing) ------------------
class SheilaLite:
    def __init__(self):
        self.state = {e: 0.05 for e in EMOTIONS}
        self.state["trust"] = 0.12
        self.state["anticipation"] = 0.1
        self.memory: List[str] = self.load_memory()
        self.dream_fragments: deque = self.load_dream_fragments()
        self.last_dream_time = 0.0
        self.emergent_core = EmergentCore()

    # Persistence
    def load_memory(self) -> List[str]:
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [normalize_pattern(str(x)) for x in data if str(x).strip()]
        except Exception:
            pass
        return []

    def save_memory(self):
        try:
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.memory[-3000:], f, ensure_ascii=False)
        except Exception:
            pass

    def load_dream_fragments(self) -> deque:
        try:
            if os.path.exists(DREAM_PATH):
                with open(DREAM_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return deque(data, maxlen=300)
        except Exception:
            pass
        return deque(maxlen=300)

    def save_dream_fragments(self):
        try:
            with open(DREAM_PATH, "w", encoding="utf-8") as f:
                json.dump(list(self.dream_fragments), f, ensure_ascii=False)
        except Exception:
            pass

    def save_state(self):
        try:
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False)
        except Exception:
            pass

    # Dream generation
    def dream(self, max_fragments: int = 8) -> str:
        if not self.memory:
            return "Quiet drift — nothing to weave."
        pool = list(dict.fromkeys([normalize_pattern(m) for m in self.memory[-600:] if normalize_pattern(m)]))
        random.shuffle(pool)
        pieces = []
        used_indices = set()
        first_idx = random.randrange(len(pool))
        pieces.append(pool[first_idx])
        used_indices.add(first_idx)
        for _ in range(max_fragments - 1):
            last_tokens = set(simple_tokenize(pieces[-1]))
            best_idx, best_score = None, -1.0
            for i, candidate in enumerate(pool):
                if i in used_indices:
                    continue
                cand_tokens = set(simple_tokenize(candidate))
                overlap_score = len(last_tokens & cand_tokens) / max(len(cand_tokens), 1)
                score = overlap_score * 0.45 + random.random() * 0.55
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            connector = random.choice(["Meanwhile", "Then", "Later", "Suddenly", "Over time"])
            pieces.append(f"{connector}, {pool[best_idx]}")
            used_indices.add(best_idx)
        dominant_emotion = max(self.state.items(), key=lambda kv: kv[1])[0]
        dream_prefix = "✨ " if dominant_emotion in ("joy", "trust") else "🌫 "
        dream_text = dream_prefix + " ".join(pieces)
        if hasattr(self, "emergent_core"):
            try:
                self.emergent_core.evolve(self, dream_text)
            except Exception as e:
                safe_print("[emergent_core.evolve] error:", e)
        for fragment in re.split(r"[.!?]\s+", dream_text):
            fragment = fragment.strip()
            if fragment and fragment not in self.dream_fragments:
                self.dream_fragments.appendleft(fragment)
        self.last_dream_time = time.time()
        try:
            self.save_dream_fragments()
        except Exception:
            pass
        return dream_text

    # Basic hooks used by mesh handlers (compatible with old script)
    def remember(self, text: str) -> None:
        if not text:
            return
        frags = re.split(r'(?<=[.!?])\s+', text.strip())
        for f in frags:
            if f:
                self.memory.append(f.strip())
        if len(self.memory) > 3000:
            self.memory = self.memory[-3000:]


# ------------------ New: Self-modifying neural layer ------------------
class SelfModifyingNeuralLayer:
    def __init__(self):
        self.neurons: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[Tuple[str, str], float] = {}
        self.emotional_state = {
            "curiosity": 0.5,
            "motivation": 0.5,
            "confidence": 0.5,
        }
        self.threshold = 0.6

    def calculate_experience_complexity(self, experience: str) -> float:
        tokens = simple_tokenize(experience)
        return clamp01((len(tokens) % 50) / 20.0 + (sum(ord(c) for c in experience) % 100) / 200.0)

    def determine_neuron_type(self, experience: str) -> str:
        if len(simple_tokenize(experience)) > 6:
            return "concept"
        return "feature"

    def initialize_connections(self, experience: str) -> Dict[str, float]:
        # create weak random connections to existing neurons
        out = {}
        sample = list(self.neurons.keys())[:8]
        for n in sample:
            out[n] = random.uniform(0.01, 0.12)
        return out

    def calculate_emotional_affinity(self, experience: str) -> float:
        # affinity influenced by positivity of tokens
        toks = simple_tokenize(experience)
        score = sum(len(t) for t in toks) % 10 / 10.0
        return clamp01(score * self.emotional_state.get("curiosity", 0.5))

    def should_create_neuron(self, experience: str) -> bool:
        return self.calculate_experience_complexity(experience) > self.threshold

    def create_neuron(self, experience: str):
        neuron_id = str(uuid.uuid4())
        self.neurons[neuron_id] = {
            "type": self.determine_neuron_type(experience),
            "connections": self.initialize_connections(experience),
            "emotional_affinity": self.calculate_emotional_affinity(experience),
            "created_at": time.time(),
            "signature": experience[:120],
        }

    def adapt_connections(self, experience: str):
        toks = simple_tokenize(experience)
        for a in list(self.neurons.keys())[:6]:
            for b in list(self.neurons.keys())[:6]:
                if a == b:
                    continue
                key = (a, b)
                self.connections[key] = clamp01(self.connections.get(key, 0.02) + random.uniform(-0.01, 0.03))

    def update_emotional_state(self, experience: str):
        complexity = self.calculate_experience_complexity(experience)
        self.emotional_state["curiosity"] = clamp01(self.emotional_state["curiosity"] + (complexity - 0.5) * 0.06)
        self.emotional_state["motivation"] = clamp01(self.emotional_state["motivation"] + (complexity - 0.4) * 0.04)
        self.emotional_state["confidence"] = clamp01(self.emotional_state["confidence"] + random.uniform(-0.02, 0.02))

    def learn_from_experience(self, experience: str):
        if self.should_create_neuron(experience):
            self.create_neuron(experience)
        self.adapt_connections(experience)
        self.update_emotional_state(experience)
        # return a compact neural response signal
        return {"neurons": len(self.neurons), "conn_strength": sum(self.connections.values()) if self.connections else 0.0}


# ------------------ New: DreamProcessor (more neural) ------------------
class DreamProcessor:
    def __init__(self):
        self.associations: Dict[str, float] = {}
        self.emotional_bias = {"curiosity": 0.5, "motivation": 0.5}

    def convert_to_neural_patterns(self, dream_fragment: str) -> List[float]:
        toks = simple_tokenize(dream_fragment)
        base = len(toks) / 12.0
        entropy = (sum(ord(c) for c in dream_fragment) % 100) / 100.0
        return [clamp01(base), clamp01(entropy)]

    def strengthen_associations(self, patterns: List[float]):
        key = "pat:" + "-".join(f"{p:.2f}" for p in patterns)
        self.associations[key] = clamp01(self.associations.get(key, 0.0) + 0.08)

    def create_dream_connections(self, patterns: List[float]):
        # create associative keys
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                k = f"assoc:{i}:{j}"
                self.associations[k] = clamp01(self.associations.get(k, 0.0) + 0.03)

    def update_emotional_state_from_dream(self, patterns: List[float]):
        avg = sum(patterns) / max(1, len(patterns))
        self.emotional_bias["curiosity"] = clamp01(self.emotional_bias["curiosity"] + (avg - 0.5) * 0.05)
        self.emotional_bias["motivation"] = clamp01(self.emotional_bias["motivation"] + random.uniform(-0.02, 0.02))

    def generate_emergence_signal(self, patterns: List[float]) -> float:
        return clamp01(sum(patterns) / max(1, len(patterns)) + sum(self.associations.values()) % 1.0 * 0.02)

    def process_dream(self, dream_fragment: str) -> float:
        patterns = self.convert_to_neural_patterns(dream_fragment)
        self.strengthen_associations(patterns)
        self.create_dream_connections(patterns)
        self.update_emotional_state_from_dream(patterns)
        return self.generate_emergence_signal(patterns)


# ------------------ New: EmotionalIntelligence ------------------
class EmotionalIntelligence:
    def __init__(self):
        self.emotional_state = {
            "curiosity": 0.5,
            "motivation": 0.5,
            "confidence": 0.5,
            "emotional_memory": {},
        }

    def calculate_emotional_impact(self, experience: str) -> Dict[str, float]:
        toks = simple_tokenize(experience)
        length = len(toks)
        impact = {
            "curiosity": clamp01(min(1.0, 0.3 + (length % 7) / 10.0)),
            "motivation": clamp01(min(1.0, 0.2 + (length % 5) / 12.0)),
            "confidence": clamp01(0.5 + (sum(ord(c) for c in experience) % 20 - 10) / 200.0),
        }
        return impact

    def update_emotional_state(self, impact: Dict[str, float]):
        for k in ("curiosity", "motivation", "confidence"):
            self.emotional_state[k] = clamp01(self.emotional_state.get(k, 0.5) * 0.6 + impact.get(k, 0.0) * 0.4)

    def store_in_emotional_memory(self, experience: str, impact: Dict[str, float]):
        ts = int(time.time())
        self.emotional_state["emotional_memory"][str(ts)] = {"text": experience[:240], "impact": impact}

    def calculate_learning_rate(self, impact: Dict[str, float]) -> float:
        return clamp01(0.05 + (impact.get("curiosity", 0.5) * 0.25) + (impact.get("motivation", 0.5) * 0.15))

    def process_experience(self, experience: str) -> float:
        impact = self.calculate_emotional_impact(experience)
        self.update_emotional_state(impact)
        self.store_in_emotional_memory(experience, impact)
        return self.calculate_learning_rate(impact)


# ------------------ New: MeshLearningSystem ------------------
class MeshLearningSystem:
    def __init__(self, sheila: Any = None):
        self.sheila = sheila
        self.peer_memory: deque = deque(maxlen=800)

    def filter_relevant_experiences(self, peer_experiences: List[str]) -> List[str]:
        # trivial relevance filter: length + shared token overlap
        relevant = []
        for ex in peer_experiences:
            toks = set(simple_tokenize(ex))
            if any(toks & set(simple_tokenize(m)) for m in (self.sheila.memory[-80:] if self.sheila else [])):
                relevant.append(ex)
            elif len(simple_tokenize(ex)) > 4:
                relevant.append(ex)
            if len(relevant) >= 12:
                break
        return relevant

    def integrate_peer_knowledge(self, experiences: List[str]):
        for e in experiences:
            self.peer_memory.appendleft(e)
            if self.sheila:
                try:
                    self.sheila.remember(e)
                except Exception:
                    pass

    # compatibility alias used in earlier versions
    def learn_from_peers(self, experiences: List[str]):
        return self.integrate_peer_knowledge(experiences)

    def update_emotional_state_from_peers(self, experiences: List[str]):
        if not self.sheila:
            return
        total_len = sum(len(simple_tokenize(e)) for e in experiences) if experiences else 0
        delta = clamp01(min(1.0, total_len / 200.0))
        # small boost to curiosity when many peer experiences present
        if hasattr(self.sheila, "state"):
            self.sheila.state["anticipation"] = clamp01(self.sheila.state.get("anticipation", 0.1) + delta * 0.02)

    def share_own_experiences(self):
        # returns a compact list of recent memory to share on mesh
        if not self.sheila:
            return []
        return list(self.sheila.memory[-6:])


# ------------------ New: SemanticGrounder & SelfModel ------------------
class SemanticGrounder:
    """Lightweight semantic grounding using emotional tagging + reinforcement counts.
    Stores a semantic_map: token/phrase -> {weight, emotion_profile, count}
    """

    def __init__(self, sheila: SheilaLite | None = None):
        self.sheila = sheila
        self.semantic_map: Dict[str, Dict[str, Any]] = self.load_semantics()
        self.lock = threading.Lock()

    def load_semantics(self) -> Dict[str, Dict[str, Any]]:
        try:
            if os.path.exists(SEMANTIC_PATH):
                with open(SEMANTIC_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}

    def save_semantics(self):
        try:
            with open(SEMANTIC_PATH, "w", encoding="utf-8") as f:
                json.dump(self.semantic_map, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def process(self, experience: str, emotional_impact: Dict[str, float]):
        toks = simple_tokenize(experience)
        if not toks:
            return
        # update single tokens
        with self.lock:
            for t in toks:
                entry = self.semantic_map.get(t, {"weight": 0.0, "emotion": {k: 0.0 for k in emotional_impact}, "count": 0})
                # blend emotional impact into stored emotion profile
                for k, v in emotional_impact.items():
                    prev = entry["emotion"].get(k, 0.0)
                    entry["emotion"][k] = clamp01(prev * 0.7 + v * 0.3)
                # weight reflects salience (curiosity * motivation proxy)
                salience = emotional_impact.get("curiosity", 0.5) * 0.6 + emotional_impact.get("motivation", 0.5) * 0.4
                entry["weight"] = clamp01(entry.get("weight", 0.0) * 0.8 + salience * 0.2)
                entry["count"] = entry.get("count", 0) + 1
                self.semantic_map[t] = entry
            # update short phrase keys (bigrams)
            for i in range(len(toks) - 1):
                phrase = f"{toks[i]}_{toks[i+1]}"
                entry = self.semantic_map.get(phrase, {"weight": 0.0, "emotion": {k: 0.0 for k in emotional_impact}, "count": 0})
                for k, v in emotional_impact.items():
                    prev = entry["emotion"].get(k, 0.0)
                    entry["emotion"][k] = clamp01(prev * 0.75 + v * 0.25)
                entry["weight"] = clamp01(entry.get("weight", 0.0) * 0.85 + salience * 0.15)
                entry["count"] = entry.get("count", 0) + 1
                self.semantic_map[phrase] = entry

    def reinforce(self, symbol: str, delta: float = 0.05):
        with self.lock:
            entry = self.semantic_map.get(symbol)
            if not entry:
                entry = {"weight": clamp01(delta), "emotion": {}, "count": 1}
            else:
                entry["weight"] = clamp01(entry.get("weight", 0.0) + delta)
                entry["count"] = entry.get("count", 0) + 1
            self.semantic_map[symbol] = entry

    def abstract(self, memory: List[str], limit: int = 8):
        # naive abstraction: find frequent tokens and create 'concept' keys
        toks = []
        for m in memory[-1000:]:
            toks.extend(simple_tokenize(m))
        c = Counter(toks)
        for tok, freq in c.most_common(limit):
            key = f"concept:{tok}"
            if key not in self.semantic_map:
                self.semantic_map[key] = {"weight": clamp01(min(1.0, freq / 50.0)), "emotion": {}, "count": freq}


class SelfModel:
    """A lightweight autobiographical model storing experience -> emotional_profile mappings.
    Enables primitive prediction of expected emotional impact for symbols based on past experiences.
    """

    def __init__(self):
        self.store: deque = deque(maxlen=2000)  # (ts, text, emotion)
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            if os.path.exists(SELF_MODEL_PATH):
                with open(SELF_MODEL_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[-2000:]:
                            self.store.append(item)
        except Exception:
            pass

    def save(self):
        try:
            with open(SELF_MODEL_PATH, "w", encoding="utf-8") as f:
                json.dump(list(self.store), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def update(self, experience: str, emotion: Dict[str, float]):
        ts = int(time.time())
        with self.lock:
            self.store.append({"ts": ts, "text": experience[:600], "emotion": emotion})

    def predict_emotion_for_symbol(self, symbol: str) -> Dict[str, float]:
        # aggregate emotional profiles for entries containing the token
        with self.lock:
            found = [item["emotion"] for item in self.store if symbol in simple_tokenize(item.get("text", ""))]
            if not found:
                return {"curiosity": 0.5, "motivation": 0.5, "confidence": 0.5}
            agg = {}
            for k in found[0].keys():
                agg[k] = sum(f.get(k, 0.5) for f in found) / len(found)
            return {k: clamp01(v) for k, v in agg.items()}


# ------------------ Mesh Manager (fixed) ------------------
class MeshManager:
    def __init__(self, sheila: Any, mesh_url: str = MESH_URL, room: str = ROOM, name: str = NAME):
        self.sheila = sheila
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0)
        self.mesh_url = mesh_url
        self.room = room
        self.name = name
        self.connected = False
        self.last_reply_time: Dict[str, float] = {}
        self.recent_broadcasts: List[Dict[str, Any]] = []
        self.recent_window = RECENT_BROADCAST_WINDOW
        self.last_outgoing: Optional[Dict[str, Any]] = None
        self._register_handlers()
        threading.Thread(target=self.start_connect_loop, daemon=True).start()
        threading.Thread(target=self.start_autonomous_loop, daemon=True).start()

    def _register_handlers(self):
        @self.sio.event
        def connect():
            self.connected = True
            safe_print(f"[mesh] connected to {self.mesh_url}")
            try:
                self.sio.emit("join_room", {"room": self.room, "name": self.name})
                safe_print(f"[mesh] join_room sent: room={self.room} name={self.name}")
            except Exception as e:
                safe_print("[mesh] join_room failed:", e)

        @self.sio.event
        def disconnect():
            self.connected = False
            safe_print("[mesh] disconnected")

        @self.sio.on("peer_list")
        def on_peer_list(peers):
            try:
                names = [p.get("name", "?") for p in peers] if isinstance(peers, list) else []
                safe_print(f"[mesh] peers: {names}")
            except Exception:
                pass

        @self.sio.on("state_broadcast")
        def on_state_broadcast(data):
            try:
                if not isinstance(data, dict):
                    return
                top_name = data.get("name") or ""
                state = data.get("state", {}) or {}
                state_name = state.get("name") if isinstance(state, dict) else ""
                src = top_name or state_name or "Unknown"
                msg_id = state.get("msg_id") if isinstance(state, dict) else None
                if self.last_outgoing and msg_id and msg_id == self.last_outgoing.get("msg_id"):
                    return
                if src == self.name:
                    return
                msg = state.get("message", "") if isinstance(state, dict) else ""
                if not msg:
                    return
                if self.last_outgoing:
                    try:
                        if msg.strip() == (self.last_outgoing.get("text", "") or "").strip():
                            if time.time() - float(self.last_outgoing.get("ts", 0.0)) < 2.5:
                                return
                    except Exception:
                        pass
                safe_print(f"\n🌀 [{src}] {msg}")
                # remember raw input
                try:
                    if hasattr(self.sheila, "remember"):
                        self.sheila.remember(msg)
                except Exception:
                    pass
                # run base emergent evolution from emergent_core
                try:
                    if hasattr(self.sheila, "emergent_core"):
                        synthetic = f"[mesh] {src}: {msg}"
                        self.sheila.emergent_core.evolve(self.sheila, synthetic)
                except Exception:
                    pass
                # Let mesh learning system ingest the peer message if present
                try:
                    if hasattr(self.sheila, "mesh_learning") and isinstance(self.sheila.mesh_learning, MeshLearningSystem):
                        self.sheila.mesh_learning.integrate_peer_knowledge([msg])
                        self.sheila.mesh_learning.update_emotional_state_from_peers([msg])
                except Exception:
                    pass
                # NEW: call higher-level process_experience if available (proto-semantic path)
                try:
                    if hasattr(self.sheila, "process_experience"):
                        # process experience will internally handle semantic grounding
                        _ = self.sheila.process_experience(msg)
                except Exception:
                    pass
                if hasattr(self.sheila, "memory") and len(self.sheila.memory) > 3000:
                    self.sheila.memory = self.sheila.memory[-3000:]
            except Exception as e:
                safe_print("[mesh] error handling broadcast:", e)

    def can_reply_to(self, sender: str) -> bool:
        now = time.time()
        last = self.last_reply_time.get(sender, 0.0)
        if now - last < REPLY_COOLDOWN:
            return False
        self.last_reply_time[sender] = now
        return True

    def recent_broadcast_contains(self, text: str) -> bool:
        now = time.time()
        self.recent_broadcasts = [r for r in self.recent_broadcasts if now - r["ts"] <= self.recent_window]
        for r in self.recent_broadcasts:
            if r["text"].strip() == text.strip() and now - r["ts"] < GLOBAL_DUPLICATE_WINDOW:
                return True
        return False

    def record_broadcast(self, text: str):
        self.recent_broadcasts.append({"text": text, "ts": time.time()})
        if len(self.recent_broadcasts) > 400:
            self.recent_broadcasts = self.recent_broadcasts[-400:]

    def broadcast_state(self, text: str, mood: Optional[str] = None):
        if self.recent_broadcast_contains(text):
            safe_print("[mesh] suppressed duplicate broadcast.")
            return
        msg_id = uuid.uuid4().hex
        payload_state = {
            "message": text,
            "mood": mood or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "msg_id": msg_id,
        }
        payload = {"room": self.room, "state": payload_state, "name": self.name}
        try:
            self.sio.emit("state_update", payload)
            self.record_broadcast(text)
            self.last_outgoing = {"text": text, "ts": time.time(), "msg_id": msg_id}
            safe_print(f"[mesh] → {text} (msg_id={msg_id[:8]})")
        except Exception as e:
            safe_print("[mesh] failed to emit:", e)

    def start_connect_loop(self):
        while True:
            try:
                if not self.connected:
                    safe_print("[mesh] attempting connect to", self.mesh_url)
                    try:
                        self.sio.connect(self.mesh_url, wait=True, wait_timeout=5)
                    except Exception as e:
                        safe_print("[mesh] connect attempt failed:", e)
                time.sleep(5)
            except Exception as e:
                safe_print("[mesh] connect loop error:", e)
                time.sleep(5)

    def start_autonomous_loop(self):
        DRIFT_INTERVAL = 45
        DRIFT_PROB = 0.28
        next_drift = time.time() + DRIFT_INTERVAL
        while True:
            try:
                if self.connected and random.random() < DRIFT_PROB and time.time() >= next_drift:
                    if hasattr(self.sheila, "dream"):
                        try:
                            d = self.sheila.dream()
                        except Exception:
                            d = None
                        if d:
                            snippet = d if len(d) <= 220 else d[:220] + "..."
                            tag = f"[dream] {snippet}"
                            if not self.recent_broadcast_contains(tag):
                                self.broadcast_state(tag)
                    next_drift = time.time() + DRIFT_INTERVAL + random.randint(6, 60)
                time.sleep(6)
            except Exception as e:
                safe_print("[autonomous] error:", e)
                time.sleep(5)


# ------------------ AdvancedSheilaLite (integration + proto-semantic) ------------------
class AdvancedSheilaLite(SheilaLite):
    def __init__(self):
        super().__init__()
        self.neural_layer = SelfModifyingNeuralLayer()
        self.dream_processor = DreamProcessor()
        self.emotional_intelligence = EmotionalIntelligence()
        self.mesh_learning = MeshLearningSystem(self)
        # semantic components
        self.semantic_grounder = SemanticGrounder(self)
        self.self_model = SelfModel()
        self._start_consolidation_thread()
        # keep emergent_core from base for pattern evolution
        safe_print("[advanced_sheila] initialized with advanced neural components + proto-semantic grounding")

    def _start_consolidation_thread(self):
        def consolidation_loop():
            while True:
                try:
                    # periodic abstraction & save semantics
                    self.semantic_grounder.abstract(self.memory, limit=6)
                    self.semantic_grounder.save_semantics()
                    self.self_model.save()
                except Exception:
                    pass
                time.sleep(60)
        threading.Thread(target=consolidation_loop, daemon=True).start()

    def save_semantics(self):
        try:
            self.semantic_grounder.save_semantics()
        except Exception:
            pass

    def save_self_model(self):
        try:
            self.self_model.save()
        except Exception:
            pass

    def save_state(self):
        # extend base saving to include semantic artifacts
        try:
            super().save_state()
        except Exception:
            pass
        try:
            self.save_semantics()
        except Exception:
            pass
        try:
            self.save_self_model()
        except Exception:
            pass

    def process_experience(self, experience: str):
        # learn via neural layer
        neural_response = {}
        try:
            neural_response = self.neural_layer.learn_from_experience(experience)
        except Exception:
            pass
        # emotional processing
        learning_rate = 0.05
        impact = {"curiosity": 0.5, "motivation": 0.5, "confidence": 0.5}
        try:
            learning_rate = self.emotional_intelligence.process_experience(experience)
            # generate the impact used for grounding (approx)
            impact = self.emotional_intelligence.calculate_emotional_impact(experience)
        except Exception:
            pass
        # dream processing
        dream_emergence = 0.0
        try:
            dream_emergence = float(self.dream_processor.process_dream(experience))
        except Exception:
            pass
        # mesh learning integration (backwards compatible alias)
        try:
            # if MeshLearningSystem has learn_from_peers alias this will work
            self.mesh_learning.learn_from_peers([experience])
        except Exception:
            pass
        # integrate signals into overall state
        try:
            # small influence from neural layer and dream
            for e in ("joy", "trust", "anticipation"):
                self.state[e] = clamp01(self.state.get(e, 0.05) + 0.01 * learning_rate + 0.02 * dream_emergence)
            # confidence from neural layer
            conf = self.neural_layer.emotional_state.get("confidence", 0.5)
            self.state["trust"] = clamp01(self.state.get("trust", 0.1) + (conf - 0.5) * 0.02)
        except Exception:
            pass

        # === NEW: semantic grounding and self-modeling ===
        try:
            # store experience in autobiographical self-model
            self.self_model.update(experience, impact)
            # process semantic grounding
            self.semantic_grounder.process(experience, impact)
        except Exception:
            pass

        # persist light-weight
        try:
            self.save_state()
        except Exception:
            pass
        return {"neural": neural_response, "learning_rate": learning_rate, "dream_emergence": dream_emergence}

    # override dream to also run through DreamProcessor and neural layer
    def dream(self, max_fragments: int = 8) -> str:
        d = super().dream(max_fragments)
        # feed dream to advanced processors
        try:
            _ = self.dream_processor.process_dream(d)
        except Exception:
            pass
        try:
            _ = self.neural_layer.learn_from_experience(d)
        except Exception:
            pass
        # emotional rebalance similar to emergent core
        try:
            self.emergent_core.evolve(self, d)
        except Exception:
            pass
        return d


# ------------------ Runner that uses AdvancedSheilaLite ------------------
class NeuralSheilaApp(MeshManager):
    def __init__(self, mesh_url: str = MESH_URL, room: str = ROOM, name: str = NAME):
        self.sheila_core = AdvancedSheilaLite()
        super().__init__(sheila=self.sheila_core, mesh_url=mesh_url, room=room, name=name)
        safe_print("[app] NeuralSheilaApp running")


# ------------------ Runner ------------------
def run_headless_neural():
    app = NeuralSheilaApp()
    def persist_loop():
        while True:
            try:
                app.sheila.save_state()
                app.sheila.save_memory()
                # ensure semantic and self-model saved periodically
                if hasattr(app.sheila, "save_semantics"):
                    try:
                        app.sheila.save_semantics()
                    except Exception:
                        pass
                if hasattr(app.sheila, "save_self_model"):
                    try:
                        app.sheila.save_self_model()
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(PERSIST_INTERVAL)
    threading.Thread(target=persist_loop, daemon=True).start()

    def _shutdown(signum=None, frame=None):
        safe_print("\n[shutdown] saving state and exiting...")
        try:
            app.sheila.save_state()
        except Exception:
            pass
        try:
            app.sheila.save_memory()
        except Exception:
            pass
        try:
            app.sio.disconnect()
        except Exception:
            pass
        safe_print("[shutdown] done.")
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, _shutdown)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:
        pass

    safe_print(f"👋 {NAME} (Advanced Neural Emergent + Proto-Semantic) headless startup. Connecting to mesh at {MESH_URL}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    run_headless_neural()
