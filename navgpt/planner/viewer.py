"""Results viewer — stdlib only, no build step.

Reads what a run already wrote (``summary.json``, ``episode_N.jsonl``,
``live/epNNNN/*``) and serves it as HTML. Read-only: it never touches the
navigation state, so it needs no habitat, GPU, or API key.

    navgpt view --port 8080
    navgpt view --run <run_name>

Three pages, one level each:

    /                              runs — one row per run, live ones auto-refresh
    /run/<name>                    the run — aggregate, tool usage, episode table
    /run/<name>/ep/<i>             the episode — trajectory, tool surface, turn-by-turn
                                   flow with the frames the agent saw (navgpt.planner.flowview)
    /run/<name>/frame/<i>/<file>   a frame, ``?w=N`` for a cached JPEG thumbnail

Links are root-relative WITHOUT a leading slash and every page carries a ``<base>``
pointing at the app root, so the viewer works both bare and behind a path-prefixed
proxy (e.g. ``/proxy/8080/``).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import flowview
from .runner import is_scored
from .turns import recorded_turns, TURN_UNIT

log = logging.getLogger("navgpt.planner.viewer")

OUTPUT_ROOT = Path("outputs")


# ── reading run artifacts ──


def list_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(root.iterdir() if root.is_dir() else [],
                    key=lambda p: p.stat().st_mtime, reverse=True):
        summary = d / "summary.json"
        if not summary.is_file():
            continue
        try:
            s = json.loads(summary.read_text())
        except ValueError:
            continue
        runs.append({
            "name": d.name,
            "aggregate": s.get("aggregate") or {},
            "scored": s.get("scored_count"),
            "excluded": s.get("excluded_count"),
            "config": s.get("config") or {},
            "episodes": s.get("episodes") or [],
            "mtime": summary.stat().st_mtime,
        })
    return runs


def load_run(root: Path, name: str) -> dict | None:
    path = root / name / "summary.json"
    if not path.is_file():
        return None
    try:
        s = json.loads(path.read_text())
    except ValueError:
        return None
    s["name"] = name
    s["scored"] = s.get("scored_count")
    s["excluded"] = s.get("excluded_count")
    s["mtime"] = path.stat().st_mtime
    return s


def frame_dir(root: Path, name: str, index: int) -> Path:
    return root / name / "live" / "ep{:04d}".format(index)


def _ctype(path: Path) -> str:
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")


def thumbnail(path: Path, width: int) -> tuple[bytes, str]:
    """Downscaled frame, cached on disk beside the original (JPEG q82)."""
    try:
        from PIL import Image
    except ImportError:
        return path.read_bytes(), _ctype(path)
    cache = path.parent / ".thumbs"
    out = cache / "{}_{}.jpg".format(path.stem, width)
    try:
        if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
            return out.read_bytes(), "image/jpeg"
        cache.mkdir(exist_ok=True)
        im = Image.open(path).convert("RGB")
        im.thumbnail((width, width), Image.LANCZOS)
        im.save(out, format="JPEG", quality=82, optimize=True, progressive=True)
        return out.read_bytes(), "image/jpeg"
    except Exception:  # noqa: BLE001 - a cache miss must never break the page
        return path.read_bytes(), _ctype(path)


_TOOL_CACHE: dict[tuple, tuple] = {}


def run_tool_usage(root: Path, run: dict) -> tuple[Counter, Counter]:
    """(calls per tool, episodes-using per tool) over the run, cached by progress."""
    key = (run["name"], len(run.get("episodes") or []), run.get("mtime"))
    hit = _TOOL_CACHE.get(key)
    if hit:
        return hit
    calls: Counter = Counter()
    eps_using: Counter = Counter()
    for e in run.get("episodes") or []:
        p = root / run["name"] / "episode_{}.jsonl".format(e.get("index"))
        if not p.is_file():
            continue
        seen = set()
        for line in p.read_text().splitlines():
            if '"tool_use"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") != "tool_use":
                continue
            raw_name = str(r.get("name") or "").replace("mcp__env__", "")
            calls[raw_name] += 1
            seen.add(raw_name)
        for t in seen:
            eps_using[t] += 1
    _TOOL_CACHE.clear()
    _TOOL_CACHE[key] = (calls, eps_using)
    return calls, eps_using


# ── helpers ──


def expected_episodes(cfg: dict) -> int:
    """How many episodes a run was asked for, from its recorded config."""
    spec = str(cfg.get("episodes", "")).strip().lower()
    total = (cfg.get("dataset_info") or {}).get("episode_count") or 0
    indices = list(range(total)) if spec == "all" else parse_episode_spec(spec)
    shard = cfg.get("shard")
    return len(indices[shard[0]::shard[1]]) if shard else len(indices)


def parse_episode_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        elif part:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def progress_of(run: dict) -> tuple[int, int]:
    done = len(run.get("episodes") or [])
    expected = expected_episodes(run.get("config") or {})
    return done, max(expected, done)


def fmt(v, nd=3):
    if v is None:
        return "&ndash;"
    if isinstance(v, float):
        return "{:.{}f}".format(v, nd).rstrip("0").rstrip(".") or "0"
    return html.escape(str(v))


def pct(v):
    return "&ndash;" if v is None else "{:.0f}%".format(100.0 * float(v))


def verdict(ep: dict) -> str:
    """The runner's scoring rule (runner.is_scored) as a pill: a turn-exhausted
    episode is a real failure; errors, refusals and invalid study episodes are
    excluded, not counted."""
    if not is_scored(ep):
        return '<span class="pill warn">excluded</span>'
    m = ep.get("metrics") or {}
    return ('<span class="pill ok">success</span>' if float(m["success"]) >= 1.0
            else '<span class="pill bad">fail</span>')


def ago(ts: float | None) -> str:
    if not ts:
        return ""
    s = max(0, int(time.time() - ts))
    if s < 90:
        return "{}s ago".format(s)
    if s < 5400:
        return "{}m ago".format(s // 60)
    if s < 172800:
        return "{}h ago".format(s // 3600)
    return "{}d ago".format(s // 86400)


def stat_cards(items) -> str:
    return "<div class=cards>{}</div>".format("".join(
        "<div class=card><div class=v>{}</div><div class=l>{}</div></div>".format(v, k)
        for k, v in items))


# ── page shell ──


def page(title: str, body: str, crumbs: list[tuple[str, str | None]], depth: int = 0,
         refresh: int = 0, wide: bool = False) -> bytes:
    """``depth`` = directory levels between this page and the app root, so that
    ``<base>`` resolves root-relative links (``run/x/ep/3``) correctly everywhere:
    ``/`` → 0, ``/run/<name>`` → 1, ``/run/<name>/ep/<i>`` → 3."""
    base = "./" if depth <= 0 else "../" * depth
    crumb_html = " <span class=sep>/</span> ".join(
        "<a href='{}'>{}</a>".format(href, html.escape(text)) if href else
        "<b>{}</b>".format(html.escape(text)) for text, href in crumbs)
    live_note = ("<span class=live>&#9679; live &middot; refreshing every {}s</span>"
                 .format(refresh) if refresh else "")
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<base href='{base}'><title>{t}</title><style>{fcss}{css}</style></head><body>"
        "<header><span class=brand>NavGPT-3</span><nav class=crumbs>{crumbs}</nav>"
        "{live}</header>"
        "<main class='{cls}'>{body}</main>"
        "<footer>read-only view of <code>outputs/</code> &middot; no environment "
        "service, GPU or API key needed</footer>"
        "<div id=lb><img alt=''></div>"
        "<script>{js}{fjs}</script>"
        "</body></html>".format(t=html.escape(title), fcss=flowview.CSS, css=CSS,
                                crumbs=crumb_html, body=body, live=live_note, base=base,
                                cls="wide" if wide else "", js=LIVE_JS % refresh,
                                fjs=flowview.JS)
    ).encode("utf-8")


# ── pages ──


def render_index(runs: list[dict]) -> bytes:
    if not runs:
        return page("NavGPT-3 runs", "<p class=muted>No runs found under "
                    "<code>{}</code>.</p>".format(html.escape(str(OUTPUT_ROOT))), [("runs", None)])
    rows = []
    live_n = 0
    for r in runs:
        a = r["aggregate"]
        cfg = r["config"]
        done, expected = progress_of(r)
        live = done < expected
        live_n += live
        spent = sum(float((e.get("agent") or {}).get("cost_usd") or 0.0) for e in r["episodes"])
        rows.append(
            "<tr><td><a class=name href='run/{n}'>{n}</a>{dot}"
            "<div class=sub>{cond} &middot; {model} &middot; {split}</div></td>"
            "<td class=num>{prog}<div class=prog><div class=bar style='width:{p:.1f}%'></div></div></td>"
            "<td class=num>{sr}</td><td class=num>{spl}</td><td class=num>{osr}</td>"
            "<td class=num>{ndtw}</td><td class=num>{excl}</td><td class=num>{cost}</td>"
            "<td class=muted>{when}</td></tr>".format(
                n=html.escape(r["name"]),
                dot=" <span class='pill ok'>live</span>" if live else "",
                cond=html.escape(str(cfg.get("condition", "?"))),
                model=html.escape(str(cfg.get("model", "?"))),
                split=html.escape(str(cfg.get("split", "?"))),
                prog="{} / {}".format(done, expected), p=100.0 * done / max(1, expected),
                sr=pct(a.get("success")), spl=pct(a.get("spl")),
                osr=pct(a.get("oracle_success")), ndtw=pct(a.get("ndtw")),
                excl=fmt(r.get("excluded")),
                cost="${:.2f}".format(spent) if spent else "&ndash;",
                when=ago(r.get("mtime")),
            )
        )
    body = (
        "<h2>Runs</h2>"
        + stat_cards([("runs", len(runs)), ("live", live_n),
                      ("episodes", sum(len(r["episodes"]) for r in runs))])
        + "<table class=runs><thead><tr><th>run</th><th class=num>episodes</th>"
          "<th class=num>SR</th><th class=num>SPL</th><th class=num>OSR</th>"
          "<th class=num>nDTW</th><th class=num>excluded</th><th class=num>list cost</th>"
          "<th>updated</th></tr></thead><tbody>{}</tbody></table>".format("".join(rows))
    )
    return page("NavGPT-3 runs", body, [("runs", None)], refresh=30 if live_n else 0)


def render_run(root: Path, run: dict) -> bytes:
    a = run.get("aggregate") or {}
    cfg = run.get("config") or {}
    done, expected = progress_of(run)
    live = done < expected
    spent = sum(float((e.get("agent") or {}).get("cost_usd") or 0.0)
                for e in (run.get("episodes") or []))
    turns_by_index = {}
    for e in run.get('episodes') or []:
        ag = e.get('agent') or {}
        source = Path(e['source_summary']).parent if e.get('source_summary') else root / run['name']
        value = ag.get('turns') if ag.get('turn_unit') == TURN_UNIT else recorded_turns(
            source / ('episode_{}.jsonl'.format(e['index'])), e.get('metrics'))
        turns_by_index[e['index']] = value
    turns = [v for v in turns_by_index.values() if v is not None]

    cards = stat_cards([
        ("SR", pct(a.get("success"))), ("SPL", pct(a.get("spl"))),
        ("OSR", pct(a.get("oracle_success"))), ("nDTW", pct(a.get("ndtw"))),
        ("NE (m)", fmt(a.get("distance_to_goal"), 2)), ("TL (m)", fmt(a.get("path_length"), 1)),
        ("steps / ep", fmt(a.get("steps_taken"), 1)),
        ("turns / ep", fmt(sum(turns) / len(turns), 1) if turns else "&ndash;"),
        ("scored", "{} <span class=muted>/ {} excl.</span>".format(
            fmt(run.get("scored")), fmt(run.get("excluded")))),
        ("list cost", "${:.2f}".format(spent) if spent else "&ndash;"),
    ])

    calls, eps_using = run_tool_usage(root, run)
    n_eps = max(1, len(run.get("episodes") or []))
    trows = []
    recorded = {}
    for k in list(calls):
        if isinstance(k, tuple):
            recorded.setdefault(k[1], set()).add(k[2]); del calls[k]
    for t, c in calls.most_common():
        ct = t
        cost, fam, what = flowview.TOOL_INFO.get(ct, ("?", "?", ""))
        shown = html.escape(t) + ("<small class=muted> recorded as {}</small>".format(
            html.escape(", ".join(sorted(recorded[t])))) if t in recorded else "")
        trows.append(
            "<tr><td><span class=dot style='background:{col}'></span><code>{t}</code></td>"
            "<td class=muted>{fam}</td><td class=num>{c}</td><td class=num>{per:.1f}</td>"
            "<td class=num>{eps}</td><td class=muted>{cost}</td></tr>".format(
                col=flowview.TOOL_COLOUR.get(ct, "#888"), t=shown, fam=fam, c=c,
                per=c / n_eps, eps=pct(eps_using[t] / n_eps), cost=html.escape(cost)))
    tools_html = (
        "<section class=panel><h3>tool usage over {n} episode{s}</h3>"
        "<table class=tools><thead><tr><th>tool</th><th>family</th><th class=num>calls</th>"
        "<th class=num>per episode</th><th class=num>episodes using</th><th>cost</th></tr>"
        "</thead><tbody>{rows}</tbody></table></section>".format(
            n=n_eps, s="" if n_eps == 1 else "s", rows="".join(trows) or
            "<tr><td colspan=6 class=muted>no tool calls recorded</td></tr>")
    ) if calls else ""

    rows = []
    for ep in run.get("episodes") or []:
        m = ep.get("metrics") or {}
        ag = ep.get("agent") or {}
        rows.append(
            "<tr><td><a class=name href='run/{r}/ep/{i}'>ep {i}</a></td><td>{v}</td>"
            "<td class=num>{ne}</td><td class=num>{spl}</td><td class=num>{ndtw}</td>"
            "<td class=num>{tl}</td><td class=num>{steps}</td><td class=num>{turns}</td>"
            "<td class=num>{cost}</td><td class=instrcell>{instr}</td></tr>".format(
                r=html.escape(run["name"]), i=ep.get("index"), v=verdict(ep),
                ne=fmt(m.get("distance_to_goal"), 2), spl=fmt(m.get("spl"), 2),
                ndtw=fmt(m.get("ndtw"), 2), tl=fmt(m.get("path_length"), 1),
                steps=fmt(ag.get("env_steps")), turns=fmt(turns_by_index.get(ep.get('index'))),
                cost="${:.2f}".format(ag["cost_usd"]) if ag.get("cost_usd") else "&ndash;",
                instr=html.escape((ep.get("instruction") or "")[:110]),
            )
        )
    body = (
        "<h2>{name}{live}</h2>"
        "<p class=sub>{cond} &middot; {model} &middot; split {split} &middot; "
        "max_turns {mt} &middot; budget ${bud}/ep &middot; {done} / {exp} episodes</p>"
        "<div class=prog><div class=bar style='width:{p:.1f}%'></div></div>"
        "{cards}{tools}"
        "<section class=panel><h3>episodes</h3>"
        "<table class=eps><thead><tr><th>episode</th><th>outcome</th><th class=num>NE m</th>"
        "<th class=num>SPL</th><th class=num>nDTW</th><th class=num>TL m</th>"
        "<th class=num>steps</th><th class=num>turns</th><th class=num>cost</th>"
        "<th>instruction</th></tr></thead><tbody>{rows}</tbody></table></section>".format(
            name=html.escape(run["name"]),
            live=" <span class='pill ok'>live</span>" if live else "",
            cond=html.escape(str(cfg.get("condition", "?"))),
            model=html.escape(str(cfg.get("model", "?"))),
            split=html.escape(str(cfg.get("split", "?"))),
            mt=fmt(cfg.get("max_turns")), bud=fmt(cfg.get("max_budget_usd"), 1),
            done=done, exp=expected, p=100.0 * done / max(1, expected),
            cards=cards, tools=tools_html, rows="".join(rows),
        )
    )
    return page(run["name"], body, [("runs", "./"), (run["name"], None)], depth=1,
                refresh=20 if live else 0)


def render_episode(root: Path, run: dict, index: int) -> bytes:
    """The episode: trajectory + tool surface + turn-by-turn flow with frames."""
    r = html.escape(run["name"])
    crumbs = [("runs", "./"), (run["name"], "run/" + r), ("ep {}".format(index), None)]
    ep_path = root / run["name"] / "episode_{}.jsonl".format(index)
    if not ep_path.is_file():
        return page("no such episode", "<p class=muted>no episode_{}.jsonl yet</p>".format(index),
                    crumbs, depth=3)
    ep = flowview.load_episode(ep_path)
    fdir = frame_dir(root, run["name"], index)
    flowview.attach_frames(ep, fdir if fdir.is_dir() else None)

    def src(shot, _ep):
        return "run/{}/frame/{}/{}?w={}".format(r, index, html.escape(shot["file"]),
                                               470 if shot.get("is_map") else 300)

    body = flowview.episode_section(ep, src)
    # click → the full-size frame (zoom() prefers data-full over src)
    body = re.sub(r'<img loading="lazy" src="(run/[^"?]+)\?w=(\d+)"',
                  r'<img loading="lazy" data-full="\1" src="\1?w=\2"', body)

    eps = [e.get("index") for e in (run.get("episodes") or [])]
    rec_of = {e.get("index"): e for e in (run.get("episodes") or [])}
    nav = "".join(
        "<option value='{i}'{sel}>ep {i} — {v}</option>".format(
            i=i, sel=" selected" if i == index else "",
            v=re.sub("<[^>]+>", "", verdict(rec_of.get(i, {}))))
        for i in eps)
    pos = eps.index(index) if index in eps else -1
    prev_i = eps[pos - 1] if pos > 0 else None
    next_i = eps[pos + 1] if 0 <= pos < len(eps) - 1 else None
    switcher = (
        "<div class=switch>"
        + ("<a href='run/{r}/ep/{i}'>&larr; ep {i}</a>".format(r=r, i=prev_i)
           if prev_i is not None else "<span class=muted>&larr;</span>")
        + " <select onchange=\"location.href='run/{r}/ep/'+this.value\">{nav}</select> "
          .format(r=r, nav=nav)
        + ("<a href='run/{r}/ep/{i}'>ep {i} &rarr;</a>".format(r=r, i=next_i)
           if next_i is not None else "<span class=muted>&rarr;</span>")
        + "</div>"
    )
    in_progress = not (ep.get("metrics") or {})
    return page("{} · ep {}".format(run["name"], index), switcher + body, crumbs,
                depth=3, refresh=10 if in_progress else 0, wide=True)


# ── server ──


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root = OUTPUT_ROOT

    def log_message(self, fmt_, *args):
        log.debug("%s - %s", self.address_string(), fmt_ % args)

    def _send(self, blob: bytes, ctype="text/html; charset=utf-8", status=200, extra=None):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(blob)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(blob)

    def _404(self, msg="not found"):
        self._send(page("not found", "<p class=muted>{}</p>".format(html.escape(msg)),
                        [("runs", "./")]), status=404)

    def do_GET(self):
        raw = self.path
        path = raw.split("?")[0].rstrip("/") or "/"
        query = raw.split("?", 1)[1] if "?" in raw else ""

        if path == "/":
            self._send(render_index(list_runs(self.root)))
            return

        m = re.match(r"^/run/([^/]+)$", path)
        if m:
            run = load_run(self.root, m.group(1))
            self._send(render_run(self.root, run)) if run else self._404("no such run")
            return

        m = re.match(r"^/run/([^/]+)/ep/(\d+)$", path)
        if m:
            run = load_run(self.root, m.group(1))
            if not run:
                self._404("no such run")
                return
            self._send(render_episode(self.root, run, int(m.group(2))))
            return

        m = re.match(r"^/run/([^/]+)/frame/(\d+)/([A-Za-z0-9_.-]+\.(?:png|jpe?g|webp))$", path)
        if m:
            name, index, fname = m.group(1), int(m.group(2)), m.group(3)
            base = frame_dir(self.root, name, index).resolve()
            target = (base / fname).resolve()
            if not str(target).startswith(str(base) + os.sep):
                self._404("no such frame")
                return
            if not target.is_file():
                self._404("no such frame")
                return
            m_w = re.search(r"(?:^|&)w=(\d{2,4})(?:&|$)", query)
            if m_w:
                blob, ctype = thumbnail(target, int(m_w.group(1)))
            else:
                blob, ctype = target.read_bytes(), _ctype(target)
            self._send(blob, ctype=ctype, extra={"cache-control": "max-age=86400"})
            return

        self._404()


CSS = """
:root{--bg:#0a0b0e;--panel:#0e1116;--line:#1a1e26;--fg:#d7dae0;--muted:#7b8290;
 --ok:#3ddc63;--bad:#ff5f56;--warn:#ffb454;--accent:#4da3ff}
body{font:13.5px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
 background:var(--bg);color:var(--fg)}
code,pre,.args,.kv,.tl,.prow,select{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:12px 22px}
.brand{font-weight:700;letter-spacing:.04em;color:#fff}
.crumbs{color:var(--muted);font-size:13px}
.crumbs a{color:var(--muted)}.crumbs a:hover{color:#fff}.crumbs b{color:#fff;font-weight:600}
.sep{color:#3a3f4a;margin:0 4px}
.live{margin-left:auto;color:var(--ok);font-size:12px}
main{padding:18px 22px 40px;max-width:1180px}
main.wide{max-width:none;padding:0 0 40px}
h2{font-size:18px;margin:6px 0 4px;color:#fff;font-weight:600}
.sub{color:var(--muted);margin:0 0 10px;font-size:12.5px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.muted{color:var(--muted)}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
 padding:10px 14px;min-width:96px}
.card .v{font-size:20px;color:#fff;font-variant-numeric:tabular-nums}
.card .l{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
section.panel{margin:0 0 16px}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
 font-variant-numeric:tabular-nums;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
th.num,td.num{text-align:right;white-space:nowrap}
tbody tr:hover td{background:#0f1319}
td .name{font-weight:600}
td .sub{margin:2px 0 0;font-size:11.5px}
td.instrcell{color:var(--muted);font-size:12px;max-width:520px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;
 border:1px solid var(--line);color:var(--fg);font-weight:600;background:transparent}
.pill.ok{color:var(--ok);border-color:#1f6f32}.pill.bad{color:var(--bad);border-color:#7d2b26}
.pill.warn{color:var(--warn);border-color:#7a5a12}
.prog{height:4px;background:#161a21;border-radius:2px;overflow:hidden;margin:4px 0 2px}
.prog .bar{height:100%;background:var(--accent)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
.switch{display:flex;align-items:center;gap:12px;padding:12px 20px 0;color:var(--muted)}
.switch select{max-width:60vw}
footer{padding:14px 22px;color:#3a3f4a;font-size:11.5px;border-top:1px solid var(--line)}
"""

LIVE_JS = """
var R=%d; if(R){setTimeout(function(){var y=window.scrollY;
 sessionStorage.setItem('y',y); location.reload();},R*1000);
 var y=sessionStorage.getItem('y'); if(y){window.scrollTo(0,parseInt(y,10));
 sessionStorage.removeItem('y');}}
"""


def build_parser():
    p = argparse.ArgumentParser(description="Browse NavGPT-3 run results")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--run", default=None, help="print the deep link for this run on startup")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def serve(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(args.output_root).resolve()
    Handler.root = root
    runs = list_runs(root)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    base = "http://{}:{}".format("127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)
    log.info("viewer on %s — %d run(s) under %s", base, len(runs), root)
    if args.run:
        log.info("  %s/run/%s", base, args.run)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
