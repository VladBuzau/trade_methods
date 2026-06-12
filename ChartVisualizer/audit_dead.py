"""
audit_dead.py — scaneaza app.py + autotrader.py si raporteaza:

  1. Chei `scanner` definite dar neaccesate
  2. Functii definite dar neapelate
  3. HTML element id-uri fara handler JS
  4. Endpoints Flask fara caller frontend (fetch/href/window.open)

Rezultat: console + audit_report.json (linia exacta a fiecarui artefact).

NU sterge nimic — doar raporteaza pentru review.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
APP_PY = HERE / "app.py"
AUTO_PY = HERE / "autotrader.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ─── 1. Chei scanner definite vs accesate ────────────────────────────────────
def audit_scanner_keys(auto_src: str) -> dict:
    """Extrage cheile din `scanner = {...}` si verifica daca sunt accesate."""
    # Gaseste blocul scanner = { ... }
    m = re.search(r"^scanner\s*=\s*\{", auto_src, re.MULTILINE)
    if not m:
        return {"error": "scanner = {...} nu a fost gasit"}
    start = m.end() - 1
    # Balanced braces parse
    depth = 0
    i = start
    while i < len(auto_src):
        if auto_src[i] == "{":
            depth += 1
        elif auto_src[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    else:
        return {"error": "Nu am putut delimita blocul scanner"}

    block = auto_src[start:end]
    # Extrage cheile top-level ("key": ...,)
    top_keys = []
    depth = 0
    for line_idx, line in enumerate(block.splitlines()):
        stripped = line.strip()
        # Numai linii la indent de top (in {})
        if depth == 1:
            km = re.match(r'^"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', stripped)
            if km:
                top_keys.append(km.group(1))
        depth += stripped.count("{") - stripped.count("}")

    # Cauta accese: scanner["key"], scanner.get("key"), scanner['key']
    access_pattern = re.compile(
        r"scanner(?:\s*\[\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"']\s*\]"
        r"|\s*\.\s*get\s*\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_]*)[\"'])"
    )
    accessed = set()
    for m in access_pattern.finditer(auto_src):
        k = m.group(1) or m.group(2)
        if k:
            accessed.add(k)

    unused = sorted(set(top_keys) - accessed)
    return {
        "total_keys": len(top_keys),
        "accessed": len(set(top_keys) & accessed),
        "unused_keys": unused,
    }


# ─── 2. Functii definite vs apelate ──────────────────────────────────────────
def _extract_def_lines(src: str) -> dict[str, int]:
    """Returneaza {func_name: line_number} pentru toate `def ...`."""
    defs = {}
    for i, line in enumerate(src.splitlines(), start=1):
        m = re.match(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line)
        if m:
            defs[m.group(1)] = i
    return defs


def audit_unused_functions(src: str, label: str) -> dict:
    defs = _extract_def_lines(src)
    # Functie e folosita daca numele apare in alta parte (apel sau referinta)
    unused = []
    for name, ln in defs.items():
        if name.startswith("__"):
            continue
        # Sare peste functii speciale Flask (route handlers etc.)
        if name in {"index", "login", "logout"} or name.startswith("_render"):
            continue
        # Cauta utilizari (oricare alt apel decat definitia)
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        hits = list(pattern.finditer(src))
        # Filtreaza definitia
        non_def_hits = [
            h for h in hits
            if not src[h.start():].startswith(f"def {name}")
        ]
        if len(non_def_hits) == 0:
            unused.append({"name": name, "line": ln})
    return {
        "file": label,
        "total_defs": len(defs),
        "unused_funcs": unused,
    }


# ─── 3. HTML element ID-uri fara handler JS ──────────────────────────────────
def audit_html_ids(src: str, label: str) -> dict:
    """Detecteaza id-uri HTML in template care nu sunt referite din JS."""
    # Caut TOATE id="..." in HTML (in stringuri Python)
    ids = re.findall(r'\bid="([a-zA-Z_][a-zA-Z0-9_\-]*)"', src)
    # Ignor id-urile dinamice (cele cu ${} sau Jinja {{ }})
    ids = [i for i in ids if "${" not in i and "{{" not in i]
    ids_set = set(ids)
    # Caut referintele JS: document.getElementById('...'), querySelector('#...')
    refs = set()
    for m in re.finditer(r"getElementById\(\s*[\"']([a-zA-Z_][a-zA-Z0-9_\-]*)[\"']\s*\)", src):
        refs.add(m.group(1))
    for m in re.finditer(r"querySelector(?:All)?\(\s*[\"']#([a-zA-Z_][a-zA-Z0-9_\-]*)[\"']\s*\)", src):
        refs.add(m.group(1))
    # Si CSS rules cu #id
    for m in re.finditer(r"#([a-zA-Z_][a-zA-Z0-9_\-]*)\s*\{", src):
        refs.add(m.group(1))
    unused = sorted(ids_set - refs)
    return {
        "file": label,
        "total_ids": len(ids_set),
        "unused_ids": unused[:50],   # primele 50 pentru raport
        "unused_count": len(unused),
    }


# ─── 4. Endpoints Flask fara caller frontend ─────────────────────────────────
def audit_endpoints(app_src: str, auto_src: str) -> dict:
    """Detecteaza endpointuri @app.route si @autotrader_bp.route fara fetch/href call."""
    combined = app_src + "\n" + auto_src
    routes = re.findall(
        r'@(?:app|autotrader_bp)\.route\(\s*["\']([^"\']+)["\']',
        combined,
    )
    # Filtreaza wildcards/path params
    simple = []
    for r in routes:
        # Inlocuieste <int:x> sau <x> cu placeholder
        canonical = re.sub(r"<[^>]+>", "X", r)
        simple.append((r, canonical))
    unused = []
    for original, canonical in simple:
        # Cauta in JS/HTML: fetch('/route'), href="/route", window.open('/route')
        # Folosesc partea statica (fara params)
        static_prefix = re.split(r"X", canonical)[0].rstrip("/")
        if not static_prefix or static_prefix == "/":
            continue
        # Caut prefixul in stringuri
        patt = re.escape(static_prefix)
        # Cauta in JS/templates
        count = 0
        for m in re.finditer(rf"[\"'`]{patt}[/'\"`?]", combined):
            count += 1
        # Daca un singur match (probabil definitia), e suspect
        # Eliminam definitia rutei insesi
        non_def = count
        if non_def <= 1:
            unused.append({"route": original, "static_prefix": static_prefix, "refs": count})
    return {
        "total_routes": len(routes),
        "unused_routes": unused,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    app_src = read(APP_PY)
    auto_src = read(AUTO_PY)

    print("=" * 70)
    print("AUDIT DEAD CODE — ChartVisualizer")
    print("=" * 70)

    report = {}

    # 1. Scanner keys
    print("\n[1/4] Scanner keys (autotrader.py)")
    print("-" * 40)
    sc = audit_scanner_keys(auto_src)
    report["scanner"] = sc
    if "error" in sc:
        print(f"  ERROR: {sc['error']}")
    else:
        print(f"  Total chei top-level: {sc['total_keys']}")
        print(f"  Accesate: {sc['accessed']}")
        print(f"  Neaccesate: {len(sc['unused_keys'])}")
        if sc["unused_keys"]:
            print("  Lista (max 20):")
            for k in sc["unused_keys"][:20]:
                print(f"    • {k}")

    # 2. Functii nefolosite
    print("\n[2/4] Functii neapelate")
    print("-" * 40)
    for src, label in [(app_src, "app.py"), (auto_src, "autotrader.py")]:
        r = audit_unused_functions(src, label)
        report[f"unused_funcs_{label}"] = r
        print(f"  {label}: {len(r['unused_funcs'])}/{r['total_defs']} neapelate")
        for fn in r["unused_funcs"][:15]:
            print(f"    • {label}:{fn['line']}  {fn['name']}()")

    # 3. HTML IDs
    print("\n[3/4] HTML id-uri fara referinta JS/CSS")
    print("-" * 40)
    for src, label in [(app_src, "app.py"), (auto_src, "autotrader.py")]:
        r = audit_html_ids(src, label)
        report[f"unused_ids_{label}"] = r
        print(f"  {label}: {r['unused_count']}/{r['total_ids']} id-uri orphan")
        for i in r["unused_ids"][:10]:
            print(f"    • #{i}")

    # 4. Endpoints fara caller
    print("\n[4/4] Endpoints Flask fara caller frontend")
    print("-" * 40)
    r = audit_endpoints(app_src, auto_src)
    report["unused_routes"] = r
    print(f"  Total endpoints: {r['total_routes']}")
    print(f"  Posibil neapelate: {len(r['unused_routes'])}")
    for ep in r["unused_routes"][:15]:
        print(f"    • {ep['route']}  (refs: {ep['refs']})")

    # Salveaza raport
    out = HERE / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n✅ Raport complet salvat in {out}")


if __name__ == "__main__":
    main()
