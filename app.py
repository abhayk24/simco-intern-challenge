"""A small web app for the cleaner choice model. Standard library only.

    python app.py            then open http://localhost:8000

Pages:
  /                     form: pick a household (or a new one), change price / message / channel
  /api/predict?...      same answer as JSON (GET with query, or POST with a JSON body)
  /runs                 every saved experiment run
  /runs/<id>            one run in full (add ?format=json for JSON)
"""

import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from explain import answer, dataset
from runlog import get_run, list_runs

STYLE = """
:root { --ink:#1d2329; --muted:#5d6670; --line:#d9dee3; --bg:#f6f7f8; --card:#fff; --accent:#1f6f5c;
        --solid:#1f6f5c; --weak:#9a6a00; --unproven:#a23b3b; }
@media (prefers-color-scheme: dark) { :root { --ink:#e6e9ec; --muted:#9aa3ad; --line:#333a41; --bg:#15191d;
        --card:#1d2227; --accent:#5cc2a5; --solid:#5cc2a5; --weak:#e0b44c; --unproven:#ef8a8a; } }
* { box-sizing:border-box } body { margin:0; font:15px/1.5 system-ui, sans-serif; color:var(--ink); background:var(--bg) }
main { max-width:1000px; margin:0 auto; padding:20px 16px 60px } h1 { font-size:22px; margin:0 0 4px }
h2 { font-size:17px; margin:24px 0 8px } a { color:var(--accent) } .muted { color:var(--muted) }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-top:14px }
form .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:10px 14px }
label { display:block; font-size:13px; color:var(--muted) } input, select { width:100%; padding:6px 8px; font:inherit;
  color:var(--ink); background:var(--bg); border:1px solid var(--line); border-radius:6px }
button { margin-top:14px; padding:8px 18px; font:inherit; color:#fff; background:var(--accent); border:0; border-radius:6px; cursor:pointer }
table { width:100%; border-collapse:collapse; font-size:14px } th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top }
th { color:var(--muted); font-weight:600 } .num { text-align:right; font-variant-numeric:tabular-nums }
.bar { height:10px; background:var(--accent); border-radius:3px } .headline { font-size:17px; font-weight:600 }
.t-solid { color:var(--solid); font-weight:600 } .t-weak { color:var(--weak); font-weight:600 } .t-not { color:var(--unproven); font-weight:600 }
.warn { color:var(--weak) } fieldset { border:1px dashed var(--line); border-radius:8px; margin:12px 0 0; padding:10px }
legend { font-size:13px; color:var(--muted) } pre { white-space:pre-wrap; word-break:break-word; font-size:13px }
"""

FIELDS = {  # name: (label, choices or None for free text, default)
    "household_id": ("Household (H001-H250, or new)", None, "new"),
    "entrant_price": ("New cleaner price ($)", None, "4.49"),
    "message": ("Shelf message", ["environment", "cleaning", "low_odor", ""], "environment"),
    "channel": ("Channel", ["online", "retail"], "online"),
    "standard_price": ("Standard price ($)", None, "3.99"),
    "budget_price": ("Budget price ($)", None, "2.99"),
    "scented_price": ("Scented price ($)", None, "4.29"),
    "use_history": ("Use recorded visits (existing household)", ["yes", "no"], "yes"),
}
NEW_FIELDS = {
    "budget": ("Budget", ["middle", "tight", "flexible"], "middle"),
    "usual_brand": ("Usual brand", ["", "standard", "budget", "scented"], ""),
    "bought_entrant_before": ("Bought the new cleaner before?", ["no", "yes"], "no"),
    "entrant_note": ("Their note after using it", ["", "good", "bad"], ""),
    "last_choice": ("Last visit choice", ["", "entrant", "standard", "budget", "scented", "none"], ""),
}


def field_html(name, spec, params):
    label, choices, default = spec
    value = params.get(name, default)
    if choices is None:
        control = f'<input name="{name}" value="{html.escape(value)}">'
    else:
        opts = "".join(f'<option value="{html.escape(c)}"{" selected" if c == value else ""}>{html.escape(c or "(none)")}</option>'
                       for c in choices)
        control = f'<select name="{name}">{opts}</select>'
    return f"<label>{html.escape(label)}{control}</label>"


def trust_class(t):
    return {"solid": "t-solid", "weak": "t-weak"}.get(t, "t-not")


def result_html(r):
    rows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{100 * v:.1f}%</td>'
        f'<td style="width:50%"><div class="bar" style="width:{min(100, 250 * v):.0f}%"></div></td></tr>'
        for k, v in r["probabilities"].items())
    reasons = "".join(
        f'<tr><td>{html.escape(x["says"])}</td><td class="num">{x["moves_entrant_chance_by_points"]:+.1f}</td>'
        f'<td class="{trust_class(x["trust"])}">{html.escape(x["trust"])}</td><td class="muted">{html.escape(x["evidence"])}</td></tr>'
        for x in r["reasons"]) or '<tr><td colspan="4" class="muted">No household-specific pieces apply.</td></tr>'
    what_if = "".join(f'<td class="num">{100 * v:.0f}%</td>' for v in r["what_if_entrant_price"].values())
    past = "".join(
        f'<tr><td>{i + 1}</td><td>{html.escape(p["choice"] or "")}</td><td class="muted">{html.escape(p["said_why"] or "")}</td>'
        f'<td class="muted">{html.escape(p["after_use_note"] or "")}</td></tr>' for i, p in enumerate(r["past_visits"]))
    warnings = "".join(f'<li class="warn">{html.escape(w)}</li>' for w in r["warnings"])
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in r["model_caveats"])
    h = r["household"]
    return f"""
<div class="card">
  <div class="headline">{html.escape(r["headline"])}</div>
  <p class="muted">Household {html.escape(h["household_id"])} · budget {html.escape(h["budget"] or "?")} ·
     usual brand {html.escape(h["usual_brand"] or "unknown")} · model {html.escape(r["model_version"])} (run {r["model_run"]}) ·
     data {html.escape(r["data_version"])}</p>
  {f"<ul>{warnings}</ul>" if warnings else ""}
  <h2>Chance of each choice</h2><table>{rows}</table>
  <h2>Why: change one thing, see how the new cleaner's chance moves</h2>
  <table><tr><th>Piece</th><th class="num">Points</th><th>Trust</th><th>Evidence</th></tr>{reasons}</table>
  <h2>What if only the new cleaner's price changed</h2>
  <table><tr><th class="num">$3.49</th><th class="num">$4.49</th><th class="num">$5.49</th></tr><tr>{what_if}</tr></table>
  {f'<h2>Past visits used as history</h2><table><tr><th>#</th><th>Choice</th><th>Said why (shown, not used)</th><th>Note after use</th></tr>{past}</table>' if past else ""}
  <h2>How far to trust this model</h2><ul>{caveats}</ul>
</div>"""


def page(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>{STYLE}</style></head><body><main>{body}</main></body></html>"""


def home(params):
    households, _ = dataset()
    result, error = "", ""
    if params:
        try:
            result = result_html(answer(params))
        except ValueError as e:
            error = f'<div class="card warn">{html.escape(str(e))}</div>'
    main_fields = "".join(field_html(n, s, params) for n, s in FIELDS.items())
    new_fields = "".join(field_html(n, s, params) for n, s in NEW_FIELDS.items())
    return page("Cleaner choice", f"""
<h1>Cleaner choice: try a household</h1>
<p class="muted">Change any input and press the button. The answer comes from the saved model (run 8).
Data is fictional. {len(households)} households in the data. <a href="/runs">See all saved runs</a> ·
<a href="/api/predict?household_id=H001">JSON example</a></p>
<form class="card" method="get" action="/">
  <div class="grid">{main_fields}</div>
  <fieldset><legend>Only used when household is "new"</legend><div class="grid">{new_fields}</div></fieldset>
  <button type="submit">Show the answer</button>
</form>{error}{result}""")


def runs_page():
    rows = "".join(
        f'<tr><td><a href="/runs/{r["id"]}">{r["id"]}</a></td><td>{html.escape(r["name"])}</td><td>{html.escape(r["status"])}</td>'
        f'<td>{html.escape(r["model_version"])}</td><td class="muted">{html.escape((r["verdict"] or "")[:160])}</td></tr>'
        for r in list_runs())
    return page("Saved runs", f"""<h1>Saved runs</h1><p class="muted"><a href="/">Back to the app</a></p>
<div class="card"><table><tr><th>ID</th><th>Run</th><th>Status</th><th>Model</th><th>Verdict (start)</th></tr>{rows}</table></div>""")


def run_page(run):
    parts = "".join(f"<h2>{html.escape(k)}</h2><pre>{html.escape(v if isinstance(v, str) else json.dumps(v, indent=2))}</pre>"
                    for k, v in run.items() if k != "name")
    return page(f"Run {run['id']}", f"""<h1>Run {run['id']}: {html.escape(run['name'])}</h1>
<p class="muted"><a href="/runs">All runs</a> · <a href="/runs/{run['id']}?format=json">JSON</a></p><div class="card">{parts}</div>""")


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, obj):
        self.send(status, json.dumps(obj, indent=2), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if url.path == "/":
                self.send(200, home(params))
            elif url.path == "/api/predict":
                self.send_json(200, answer(params))
            elif url.path == "/runs":
                self.send(200, runs_page())
            elif url.path.startswith("/runs/"):
                run = get_run(int(url.path.split("/")[2]))
                self.send_json(200, run) if params.get("format") == "json" else self.send(200, run_page(run))
            elif url.path == "/health":
                self.send_json(200, {"ok": True})
            else:
                self.send(404, page("Not found", "<h1>Not found</h1><p><a href='/'>Back</a></p>"))
        except (ValueError, KeyError) as e:
            self.send_json(400, {"error": str(e)})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            params = {k: str(v) for k, v in json.loads(self.rfile.read(length) or b"{}").items()}
            self.send_json(200, answer(params))
        except (ValueError, KeyError) as e:
            self.send_json(400, {"error": str(e)})


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"Open http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
