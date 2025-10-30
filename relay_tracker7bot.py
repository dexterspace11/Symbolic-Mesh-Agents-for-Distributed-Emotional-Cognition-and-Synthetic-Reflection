#!/usr/bin/env python3
"""
🌐 Entangled Mesh Relay 7.1 — Reliable Broadcast Fix
Ensures Alfred's state reaches all peers.
"""

from flask import Flask, request
from flask_socketio import SocketIO, join_room, emit
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# {room: {sid: {"name": str}}}
ROOMS = {}

@app.route("/")
def index():
    return "🌐 Entangled Mesh Relay 7.1 active and stable."

@socketio.on("connect")
def on_connect():
    print(f"[+] Client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    for room, cmap in list(ROOMS.items()):
        if sid in cmap:
            name = cmap[sid]["name"]
            print(f"[-] {name} left {room}")
            del cmap[sid]
            socketio.emit("peer_list", list(cmap.values()), room=room)
            if not cmap:
                del ROOMS[room]
            break

@socketio.on("join_room")
def on_join_room(data):
    room = data.get("room", "default")
    name = data.get("name", f"Guest-{request.sid[:4]}")
    join_room(room)
    cmap = ROOMS.setdefault(room, {})
    cmap[request.sid] = {"name": name}
    print(f"[room={room}] {name} joined")
    socketio.emit("peer_list", list(cmap.values()), room=room)

@socketio.on("state_update")
def on_state_update(data):
    """
    Expected: { "room": "room1", "state": { "name": "Alfred", "message": "...", ... } }
    """
    room = data.get("room", "default")
    state = data.get("state", {})
    sender = state.get("name", "Unknown")

    # Ensure sender is registered in the room
    if room not in ROOMS:
        ROOMS[room] = {}
    if request.sid not in ROOMS[room]:
        ROOMS[room][request.sid] = {"name": sender}
        join_room(room)
        print(f"[auto-join] {sender} added to {room}")

    payload = {"name": sender, "state": state, "timestamp": time.time()}
    print(f"[Broadcast] {sender} -> room={room} | {state.get('message','(no message)')}")
    
    # ✅ Broadcast to everyone (including Alfred for debug visibility)
    socketio.emit("state_broadcast", payload, room=room)

# ------------------- Run -------------------

if __name__ == "__main__":
    print("🌐 Entangled Mesh Relay 7.1 running at http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
