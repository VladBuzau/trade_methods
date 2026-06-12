"""
Launcher — porneste instante ChartVisualizer si afiseaza un dashboard unificat.
Port 5003: dashboard
Port 5004: Cont 1
Port 5005: Cont 2
"""
import subprocess
import sys
import os
import time
from flask import Flask, render_template_string

app = Flask(__name__)

INSTANCES = [
    {"label": "Cont 1", "port": 5004, "color": "#1976d2"},
    {"label": "Cont 2", "port": 5005, "color": "#26a69a"},
]

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ChartVisualizer</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0d0d0d; color:#eee; font-family:'Segoe UI',monospace; display:flex; flex-direction:column; height:100vh; overflow:hidden; }

.topbar {
    display:flex; align-items:center; gap:10px;
    background:#111; border-bottom:1px solid #1e1e1e;
    padding:7px 14px; flex-shrink:0;
}
.topbar-title { font-size:0.88rem; color:#888; font-weight:400; margin-right:4px; }
.topbar-title b { color:#eee; }

.status-dot {
    width:7px; height:7px; border-radius:50%; display:inline-block;
    margin-right:4px; background:#333; transition:background 0.3s;
}
.status-dot.online { background:#26a69a; box-shadow:0 0 4px #26a69a88; }

.account-pill {
    display:flex; align-items:center; gap:6px;
    background:#1a1a1a; border:1px solid #2a2a2a;
    border-radius:5px; padding:4px 10px; font-size:0.8rem; color:#aaa;
    cursor:pointer; transition:border-color 0.15s;
}
.account-pill:hover { border-color:#444; color:#eee; }
.account-pill.active-pill { border-color:#333; color:#eee; background:#222; }
.pill-label { font-weight:600; }
.pill-port { font-size:0.7rem; color:#555; }

.btn-add {
    background:none; border:1px dashed #333; color:#555;
    padding:4px 12px; border-radius:5px; font-size:0.8rem;
    cursor:pointer; transition:all 0.15s;
}
.btn-add:hover { border-color:#555; color:#aaa; }

.btn-remove {
    background:none; border:none; color:#444; font-size:0.9rem;
    cursor:pointer; padding:0 2px; margin-left:4px; line-height:1;
}
.btn-remove:hover { color:#ef5350; }

.spacer { flex:1; }

.frames-container { flex:1; display:flex; overflow:hidden; }

.frame-wrap {
    flex:1; display:flex; flex-direction:column;
    border-right:1px solid #1a1a1a; min-width:0;
}
.frame-wrap:last-child { border-right:none; }

.frame-header {
    display:flex; align-items:center; gap:8px;
    padding:4px 10px; background:#141414; border-bottom:1px solid #1e1e1e;
    flex-shrink:0;
}
.frame-label { font-size:0.78rem; font-weight:600; }
.frame-port  { font-size:0.7rem; color:#444; }
.frame-links { display:flex; gap:4px; margin-left:auto; }
.frame-link {
    font-size:0.73rem; color:#555; cursor:pointer; padding:2px 8px;
    border-radius:3px; background:#1a1a1a; border:none; text-decoration:none;
    transition:color 0.12s, background 0.12s;
}
.frame-link:hover { color:#aaa; background:#222; }
.frame-link.active-link { color:#eee; background:#2a2a2a; }

.open-new {
    font-size:0.7rem; color:#444; text-decoration:none; margin-left:6px;
}
.open-new:hover { color:#888; }

iframe { flex:1; border:none; width:100%; height:100%; }
</style>
</head>
<body>

<div class="topbar">
    <div class="topbar-title"><b>ChartVisualizer</b></div>

    <div id="pills" style="display:flex;gap:6px;align-items:center"></div>

    <button class="btn-add" id="btn-add" onclick="addAccount()">+ Cont nou</button>

    <div class="spacer"></div>
</div>

<div class="frames-container" id="frames"></div>

<script>
const ALL_PORTS  = {{ ports }};
const ALL_LABELS = {{ labels }};
const ALL_COLORS = {{ colors }};

// Conturi active — incepe cu primul
let active = [0];  // indexi din ALL_PORTS

function render() {
    // Pills
    const pills = document.getElementById('pills');
    pills.innerHTML = '';
    active.forEach((idx, pos) => {
        const pill = document.createElement('div');
        pill.className = 'account-pill active-pill';
        pill.innerHTML = `
            <span class="status-dot" id="dot-${idx}"></span>
            <span class="pill-label" style="color:${ALL_COLORS[idx]}">${ALL_LABELS[idx]}</span>
            <span class="pill-port">${ALL_PORTS[idx]}</span>
            ${active.length > 1 ? `<button class="btn-remove" onclick="removeAccount(${pos})" title="Ascunde">✕</button>` : ''}
        `;
        pills.appendChild(pill);
    });

    // Buton add — ascunde daca toate conturile sunt deja active
    document.getElementById('btn-add').style.display = active.length >= ALL_PORTS.length ? 'none' : '';

    // Frames
    const fc = document.getElementById('frames');
    fc.innerHTML = '';
    active.forEach((idx, pos) => {
        const existingFrame = document.getElementById('frame-' + idx);
        const currentSrc = existingFrame ? existingFrame.src : `http://localhost:${ALL_PORTS[idx]}/`;

        const wrap = document.createElement('div');
        wrap.className = 'frame-wrap';
        wrap.id = 'wrap-' + idx;
        wrap.innerHTML = `
            <div class="frame-header">
                <span class="frame-label" style="color:${ALL_COLORS[idx]}">${ALL_LABELS[idx]}</span>
                <span class="frame-port">localhost:${ALL_PORTS[idx]}</span>
                <div class="frame-links">
                    <a class="frame-link active-link" onclick="navFrame(${idx}, '/', this)">Chart</a>
                    <a class="frame-link" onclick="navFrame(${idx}, '/autotrader', this)">AutoTrader</a>
                    <a class="frame-link" onclick="navFrame(${idx}, '/trades', this)">Trades</a>
                    <a class="frame-link" onclick="navFrame(${idx}, '/account', this)">Cont</a>
                </div>
                <a class="open-new" href="http://localhost:${ALL_PORTS[idx]}/" target="_blank" title="Deschide in tab nou">↗</a>
            </div>
            <iframe id="frame-${idx}" src="${currentSrc}"></iframe>
        `;
        fc.appendChild(wrap);
    });
}

function addAccount() {
    // Adauga primul cont care nu e deja activ
    for (let i = 0; i < ALL_PORTS.length; i++) {
        if (!active.includes(i)) {
            active.push(i);
            render();
            return;
        }
    }
}

function removeAccount(pos) {
    active.splice(pos, 1);
    render();
}

function navFrame(idx, path, btn) {
    document.getElementById('frame-' + idx).src = 'http://localhost:' + ALL_PORTS[idx] + path;
    document.querySelectorAll('#wrap-' + idx + ' .frame-link').forEach(l => l.classList.remove('active-link'));
    btn.classList.add('active-link');
}

// Status ping
async function checkStatus() {
    for (let i = 0; i < ALL_PORTS.length; i++) {
        const dot = document.getElementById('dot-' + i);
        if (!dot) continue;
        try {
            const r = await fetch('http://localhost:' + ALL_PORTS[i] + '/mt5_status',
                                  {signal: AbortSignal.timeout(1500)});
            const d = await r.json();
            dot.className = (d.ok && d.connected) ? 'status-dot online' : 'status-dot';
        } catch(e) {
            dot.className = 'status-dot';
        }
    }
}

render();
checkStatus();
setInterval(checkStatus, 5000);
</script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    ports  = [i["port"]  for i in INSTANCES]
    labels = [i["label"] for i in INSTANCES]
    colors = [i["color"] for i in INSTANCES]
    return render_template_string(
        DASHBOARD_HTML,
        ports=str(ports),
        labels=str(labels),
        colors=str(colors),
    )

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    print("\n  ChartVisualizer — Launcher")
    print("  ================================")
    print(f"  Dashboard:  http://localhost:5003")
    print(f"  Cont 1:     http://localhost:5004")
    print(f"  Cont 2:     http://localhost:5005")
    print()

    p1 = subprocess.Popen(
        [sys.executable, "app.py", "5004"],
        cwd=here,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )
    print("  [OK] Cont 1 pornit (PID", p1.pid, ")")

    time.sleep(1)
    p2 = subprocess.Popen(
        [sys.executable, "app.py", "5005"],
        cwd=here,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )
    print("  [OK] Cont 2 pornit (PID", p2.pid, ")")

    time.sleep(2)
    print("\n  Dashboard pornit. Apasa Ctrl+C pentru a opri tot.\n")

    import webbrowser
    webbrowser.open("http://localhost:5003")

    try:
        app.run(host="0.0.0.0", port=5003, debug=False)
    finally:
        print("\n  Se opresc instantele...")
        p1.terminate()
        p2.terminate()
