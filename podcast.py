#!/usr/bin/env python3
"""Täglicher KI-Podcast: Quellen holen -> filtern -> ranken -> Skripte -> TTS -> RSS-Feed."""
import os, re, io, json, time, asyncio, html, hashlib, urllib.request, threading
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import quote_plus
import feedparser, yaml, edge_tts, trafilatura
from google import genai
from google.genai import types
from mutagen.mp3 import MP3
from pypdf import PdfReader

SITE = Path("site")
MONTHS = ["Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"]
WEEKDAYS = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def entry_date(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    return dt.datetime(*t[:6], tzinfo=dt.timezone.utc) if t else None


# ---------------------------------------------------------------- 1. Quellen
def fetch_candidates(cfg):
    n = cfg["arxiv_per_query"]
    base = "https://export.arxiv.org/api/query?sortBy=submittedDate&sortOrder=descending"
    sources = []
    for cat in cfg.get("arxiv_categories", []):
        sources.append((f"arXiv {cat}", f"{base}&search_query=cat:{cat}&max_results={n}", True))
    for name, q in cfg.get("arxiv_queries", {}).items():
        sources.append((f"arXiv {name}", f"{base}&search_query={quote_plus(q)}&max_results={n}", True))
    for name, url in cfg.get("feeds", {}).items():
        sources.append((name, url, False))

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=cfg["lookback_days"])
    items, seen_links = [], set()
    for name, url, is_arxiv in sources:
        try:
            feed = feedparser.parse(url, agent="Mozilla/5.0 (ki-podcast)")
        except Exception as ex:
            print(f"WARN {name}: {ex}"); continue
        if not feed.entries:
            print(f"WARN {name}: keine Einträge"); continue
        for e in feed.entries:
            link = e.get("link") or e.get("id")
            if not link or link in seen_links:
                continue
            d = entry_date(e)
            if d and d < cutoff:
                continue
            seen_links.add(link)
            title = clean(e.get("title"))
            text = clean(e.get("summary") or e.get("description"))
            items.append({
                "id": hashlib.sha1(link.encode()).hexdigest()[:12],
                "source": name, "title": title, "text": text,
                "link": link, "arxiv": is_arxiv,
            })
        if is_arxiv:
            time.sleep(3)  # arXiv-Höflichkeitspause
    print(f"{len(items)} Kandidaten aus {len(sources)} Quellen")
    return items


def _kw_regex(words):
    """Stichwortliste -> Regex. 'prosthe*' = Wortanfang, sonst ganzes Wort."""
    parts = []
    for w in words or []:
        w = w.lower().strip()
        if not w:
            continue
        parts.append(r"(?<!\w)" + (re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) + r"(?!\w)"))
    return re.compile("|".join(parts), re.I) if parts else None


def prefilter(cfg, cands):
    """Wirft offensichtlich Uninteressantes vorab raus (spart Zeit und LLM-Tokens).
    Ein Eintrag fliegt nur, wenn er ein exclude-Stichwort enthält UND kein Hardware-Stichwort."""
    ex, hw = _kw_regex(cfg.get("exclude_keywords")), _kw_regex(cfg.get("hardware_keywords"))
    if not ex:
        return cands
    out, dropped = [], []
    for c in cands:
        t = f"{c['title']} {c['text'][:500]}"
        if ex.search(t) and not (hw and hw.search(t)):
            dropped.append(c)
        else:
            out.append(c)
    print(f"Vorfilter: {len(dropped)} von {len(cands)} aussortiert, z.B.: " +
          " | ".join(c["title"][:50] for c in dropped[:4]))
    return out


def arxiv_pdf_text(link, limit):
    """Volltext eines arXiv-Papers aus dem PDF (wenige Papers pro Tag, 3 s Pause)."""
    m = re.search(r"/abs/(.+)$", link)
    if not m:
        return None
    try:
        time.sleep(3)
        req = urllib.request.Request(f"https://arxiv.org/pdf/{m.group(1)}",
                                     headers={"User-Agent": "Mozilla/5.0 (ki-podcast; private use)"})
        data = urllib.request.urlopen(req, timeout=60).read()
        reader = PdfReader(io.BytesIO(data))
        txt = re.sub(r"\s+", " ", "\n".join((pg.extract_text() or "") for pg in reader.pages[:40]))
        cut = txt.rfind("References")
        if cut > len(txt) * 0.5:
            txt = txt[:cut]
        return txt[:limit] if len(txt) > 2000 else None
    except Exception as ex:
        print(f"PDF-Fehler {link}: {str(ex)[:100]}")
        return None


def article_text(cfg, c):
    """Liefert (Text, Art der Quelle) je nach content_mode: 'abstract' oder 'fulltext'."""
    if cfg.get("content_mode") != "fulltext":
        return c["text"][:7000], "Abstract bzw. Teaser"
    limit = cfg.get("fulltext_chars", 30000)
    if c["arxiv"]:
        t = arxiv_pdf_text(c["link"], limit)
        return (t, "Volltext des Papers (evtl. gekürzt)") if t else (c["text"], "nur Abstract (Volltext nicht abrufbar)")
    try:
        raw = trafilatura.fetch_url(c["link"])
        t = trafilatura.extract(raw) if raw else None
        if t and len(t) > len(c["text"]):
            return t[:limit], "Volltext des Artikels (evtl. gekürzt)"
    except Exception:
        pass
    return c["text"], "nur Teaser (Volltext nicht abrufbar)"


# ---------------------------------------------------------------- LLM
BUDGET = {"left": 10**9}  # harte Obergrenze für LLM-Anfragen pro Tag (inkl. Wiederholungen)
MODELS = {"list": [], "i": 0}  # Modellkette aus config.yaml; wird bei Fehlern der Reihe nach durchgegangen
_llm_lock = threading.Lock()  # schützt BUDGET und MODELS bei parallelen Aufrufen


def _make_conf(cfg, json_mode, no_thinking=False):
    """Baut GenerateContentConfig mit optionalem Thinking-Budget."""
    kwargs = {}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if not no_thinking:
        level = cfg.get("thinking_level", "")
        budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}.get(level)
        if budget is not None:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            except Exception:
                pass
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def llm(client, cfg, prompt, json_mode=False, _no_thinking=False):
    conf = _make_conf(cfg, json_mode, _no_thinking)
    fails = 0
    while True:
        with _llm_lock:
            if MODELS["i"] >= len(MODELS["list"]):
                raise RuntimeError("Kein Gemini-Modell verfügbar (siehe 'models' in config.yaml)")
            if BUDGET["left"] <= 0:
                raise RuntimeError("Tagesbudget für LLM-Anfragen erreicht (daily_llm_budget)")
            BUDGET["left"] -= 1
            model = MODELS["list"][MODELS["i"]]
        try:
            r = client.models.generate_content(model=model, contents=prompt, config=conf)
            if r.text:
                return r.text
            err = "leere Antwort"
        except Exception as ex:
            err = str(ex)
            if "404" in err or "NOT_FOUND" in err:
                print(f"Modell {model} nicht verfügbar -> nächstes Modell")
                with _llm_lock:
                    if MODELS["i"] < len(MODELS["list"]) - 1:
                        MODELS["i"] += 1
                continue
            if not _no_thinking and ("thinking" in err.lower() or "INVALID_ARGUMENT" in err):
                print(f"Thinking nicht unterstützt ({model}), deaktiviere für diesen Lauf")
                return llm(client, cfg, prompt, json_mode=json_mode, _no_thinking=True)
        fails += 1
        print(f"LLM-Fehler ({model}, Versuch {fails}): {err[:150]}")
        if fails >= 6:
            raise RuntimeError("LLM nicht erreichbar")
        with _llm_lock:
            if fails % 3 == 0 and MODELS["i"] < len(MODELS["list"]) - 1:
                MODELS["i"] += 1  # Tageslimit erschöpft -> nächstes Modell
        time.sleep(15 * fails)


def _json_list(text):
    data = json.loads(text)
    if isinstance(data, dict):  # manche Modelle verpacken die Liste in ein Objekt
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------- 2. Auswahl
RANK_PROMPT = """Du bist Chefredakteur eines Technik-Podcasts für einen Ingenieur mit breitem Interesse.

HAUPTINTERESSEN (hier gibt es Bonuspunkte):
{interests}

NEBENINTERESSEN:
{interests_side}

UNINTERESSANT (Punktabzug):
{avoid}

PUNKTE (starte bei 5, maximal 10):
+2  neues physikalisches/mechanisches Prinzip, neue Hardware, neues Material oder wirklich neues Konzept
+1  echter Beleg: Prototyp, Experiment, Flug-/Wassertest, klinische oder in-vivo-Daten (nicht nur Simulation)
+1  passt zu den Hauptinteressen
+1  überraschend und auch für Nicht-Spezialisten nach kurzer Erklärung spannend
-2  reine Software/KI/Simulation ohne Hardware-Bezug
-2  Firmen-PR, Finanzierung, Meinung, Ankündigung ohne technische Substanz
-1  inkrementelle Verbesserung, Benchmark, Dataset, Review ohne neue Idee
-1  rein theoretisch ohne erkennbare Anwendung
Eine 9 oder 10 vergibst du höchstens 1-2 Mal pro Tag. Eine 7 bedeutet: lohnt sich wirklich in einer Folge.
Wenn mehrere Einträge dieselbe Geschichte behandeln, gib bei den schwächeren das Feld "dup_of" mit der Nummer des besten an.

Wähle bis zu {n_total} Einträge (die besten zuerst, auch solche knapp unter 7 für Kurzmeldungen).
Gib jedem Eintrag ein "area" aus dieser Liste: {areas}.
Antworte NUR mit JSON:
[{{"id": <Nummer>, "score": <1-10>, "area": "<Bereich>", "dup_of": <Nummer oder null>, "why": "<ein Satz>"}}]

LISTE:
{listing}"""


def rank(client, cfg, cands):
    """Liefert (Hauptthemen [(c, score, area)], Kurzmeldungen [c])."""
    mpi = cfg["minutes_per_item"]
    n_min = max(1, round((cfg["min_minutes"] - 1) / mpi))
    n_max = max(n_min, round((cfg["max_minutes"] - 1) / mpi))
    q_n = cfg.get("quick_items", 10)
    areas = cfg.get("areas") or ["Sonstiges"]
    listing = "\n".join(f"[{i}] ({c['source']}) {c['title']} — {c['text'][:350]}" for i, c in enumerate(cands))
    prompt = RANK_PROMPT.format(
        interests=cfg["interests"], interests_side=cfg.get("interests_side", "-"), avoid=cfg["avoid"],
        n_total=n_max + q_n + 5, areas=", ".join(areas), listing=listing)
    data = _json_list(llm(client, cfg, prompt, json_mode=True))

    ranked, used = [], set()
    for d in data:
        if not isinstance(d, dict):
            continue
        i, s = d.get("id"), d.get("score", 0)
        if not (isinstance(i, int) and 0 <= i < len(cands)) or i in used:
            continue
        used.add(i)
        dup = d.get("dup_of")
        ranked.append({"i": i, "c": cands[i], "s": s if isinstance(s, (int, float)) else 0,
                       "area": d.get("area") if d.get("area") in areas else "Sonstiges",
                       "dup": dup if isinstance(dup, int) else None})
    ranked.sort(key=lambda r: -r["s"])

    # Duplikate (gleiche Geschichte) entfernen: der bessere Eintrag bleibt
    kept, kept_ids = [], set()
    for r in ranked:
        if r["dup"] is not None and r["dup"] in kept_ids:
            continue
        kept.append(r); kept_ids.add(r["i"])

    # Hauptthemen: Mindestscore + Begrenzung pro Bereich und Quelle für gute Mischung
    max_area, max_src, max_arx = cfg.get("max_per_area", 4), cfg.get("max_per_source", 3), cfg.get("max_arxiv_items", 8)
    main, rest, n_area, n_src = [], [], {}, {}
    for r in kept:
        grp = "arXiv" if r["c"]["arxiv"] else r["c"]["source"]
        limit = max_arx if r["c"]["arxiv"] else max_src
        if (r["s"] >= cfg["min_score"] and len(main) < n_max
                and n_area.get(r["area"], 0) < max_area and n_src.get(grp, 0) < limit):
            main.append(r); n_area[r["area"]] = n_area.get(r["area"], 0) + 1; n_src[grp] = n_src.get(grp, 0) + 1
        else:
            rest.append(r)
    while len(main) < n_min and rest:  # Mindestlänge: mit den besten Übrigen auffüllen
        main.append(rest.pop(0))
    quick = [r["c"] for r in rest if r["s"] >= cfg.get("quick_min_score", 6)][:q_n]
    print("Bereiche:", ", ".join(f"{a} {n}" for a, n in sorted(n_area.items())) or "-")
    return [(r["c"], r["s"], r["area"]) for r in main], quick


def quick_summaries(client, cfg, items):
    """Je Kurzmeldung ca. 30 s gesprochener Text + 1 Satz für die Shownotes -> [(c, spoken, note)]."""
    if not items:
        return []
    words = int(cfg.get("quick_seconds", 30) * 2.5)
    listing = "\n".join(f"[{i}] ({c['source']}) {c['title']} — {c['text'][:700]}" for i, c in enumerate(items))
    prompt = f"""Du schreibst den Podcast-Teil "Kurzmeldungen" (Deutsch). Zu jedem Eintrag:
- "spoken": gesprochener Text, ca. {words} Wörter (ca. {cfg.get('quick_seconds', 30)} Sekunden). Erster Satz nennt das Thema klar,
  damit Hörer merken, dass ein neues beginnt. Dann: was ist neu, wie funktioniert es in einfachen Worten, warum interessant.
  Nur Fakten aus dem Eintrag, nichts erfinden, Fachbegriffe kurz erklären. Reiner Fließtext, kein Markdown.
- "note": ein Satz (max. 25 Wörter) für die Shownotes.
Antworte NUR mit JSON: [{{"id": <Nummer>, "spoken": "...", "note": "..."}}]

EINTRÄGE:
{listing}"""
    res = {}
    try:
        for d in _json_list(llm(client, cfg, prompt, json_mode=True)):
            i = d.get("id") if isinstance(d, dict) else None
            if isinstance(i, int) and 0 <= i < len(items):
                res[i] = (clean(str(d.get("spoken", ""))), clean(str(d.get("note", ""))))
    except Exception as ex:
        print(f"Kurzmeldungen nicht erzeugt: {str(ex)[:120]}")
    return [(c, *res.get(i, ("", ""))) for i, c in enumerate(items)]


def quick_lines(qs):
    lines = [("LENA", "Zum Schluss noch die Kurzmeldungen: was sonst noch los war, jeweils in etwa einer halben Minute.")]
    for n, (_, spoken, _) in enumerate(qs):
        lines.append(("JONAS" if n % 2 == 0 else "LENA", spoken))
    return lines


def build_notes(done, qs):
    li = lambda c, extra="": f'<li><a href="{esc(c["link"])}">{esc(c["title"])}</a> ({esc(c["source"])}){extra}</li>'
    h = "<p><b>Themen dieser Folge</b></p><ul>" + "".join(li(c) for c in done) + "</ul>"
    if qs:
        h += "<p><b>Kurzmeldungen (nicht ausführlich besprochen)</b></p><ul>" + "".join(
            li(c, f": {esc(note)}" if note else "") for c, _, note in qs) + "</ul>"
    return h


SEGMENT_PROMPT = """Schreibe ein Podcast-Skript auf Deutsch als Gespräch zwischen LENA (Moderatorin, neugierig, stellt Verständnisfragen)
und JONAS (Ingenieur, erklärt anschaulich). Thema {k} von {n} der heutigen Folge.

QUELLE ({source}), Art des Materials: {kind}
TITEL: {title}
{text}

ANFORDERUNGEN:
- Länge ca. {words} Wörter (das ist wichtig: nicht kürzer).
- Aufbau: (1) kurzer Aufhänger, worum es geht; (2) Grundlagen: das Basiswissen, das man zum Verständnis braucht, von Null erklärt, mit einer
  Analogie; (3) Was ist neu und wie funktioniert es konkret; (4) Grenzen, offene Fragen, ehrliche Einordnung; (5) Ausblick: wohin könnte das führen.
- Bei Volltext: gehe auch auf Methode, konkrete Ergebnisse und Einschränkungen der Autoren ein.
- Aussagen über die konkrete Arbeit / Neuigkeit NUR aus der Quelle. Keine erfundenen Zahlen oder Details. Wo die Quelle dünn ist, sage das offen
  ("laut den Autoren", "im Abstract steht ..."). Allgemeines Hintergrundwissen darfst du ergänzen, aber klar als Hintergrund kennzeichnen.
- Gesprochene Sprache, kurze Sätze, Fachbegriffe beim ersten Mal erklären. Englische Fachbegriffe sind ok.
- Beginne direkt beim Thema, KEINE Begrüßung, KEINE Verabschiedung.
- Format strikt: jede Zeile "LENA: ..." oder "JONAS: ...". Kein Markdown, keine Regieanweisungen, keine Aufzählungen."""


def segment(client, cfg, c, k, n):
    words = int(cfg["minutes_per_item"] * 150 * 1.05)
    text, kind = article_text(cfg, c)
    p = SEGMENT_PROMPT.format(k=k, n=n, source=c["source"], title=c["title"],
                              text=text, kind=kind, words=words)
    return parse_script(llm(client, cfg, p))


def parse_script(t):
    lines = []
    for ln in t.replace("*", "").splitlines():
        m = re.match(r"^\s*(LENA|JONAS)\s*:\s*(.+)$", ln.strip())
        if m:
            lines.append((m.group(1), m.group(2).strip()))
    return lines


def intro_outro(client, cfg, titles, today, quick_count=0):
    datum = f"{WEEKDAYS[today.weekday()]}, {today.day}. {MONTHS[today.month-1]}"
    extra = f"\nErwähne im Intro kurz, dass am Ende {quick_count} Kurzmeldungen folgen." if quick_count else ""
    p = f"""Schreibe Intro und Outro für den Podcast "{cfg['podcast_title']}" als Gespräch zwischen LENA und JONAS (Deutsch, locker).
Heute ist {datum}. Themen der Folge:
{chr(10).join('- ' + t for t in titles)}{extra}
INTRO: ca. 70 Wörter, Begrüßung, Datum, Vorschau auf die Themen (ohne Details zu verraten).
OUTRO: ca. 40 Wörter, kurzer Abschied.
Format strikt: zuerst eine Zeile "### INTRO", dann Zeilen "LENA: ..." / "JONAS: ...", dann "### OUTRO" und Zeilen. Kein Markdown sonst."""
    t = llm(client, cfg, p)
    a, _, b = t.partition("### OUTRO")
    return parse_script(a.replace("### INTRO", "")), parse_script(b)


# ---------------------------------------------------------------- TTS
def chunks(t, limit=700):
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", t):
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


async def synth(lines, cfg):
    sem = asyncio.Semaphore(4)

    async def one(speaker, text):
        async with sem:
            for attempt in range(5):
                voice = cfg["voices"][speaker] if attempt < 3 else cfg["fallback_voices"][speaker]
                try:
                    buf = b""
                    async for ch in edge_tts.Communicate(text, voice, rate=cfg["speech_rate"]).stream():
                        if ch["type"] == "audio":
                            buf += ch["data"]
                    if buf:
                        return buf
                except Exception as ex:
                    print(f"TTS-Fehler ({voice}): {str(ex)[:100]}")
                await asyncio.sleep(2 * (attempt + 1))
            raise RuntimeError("TTS fehlgeschlagen")

    jobs = [one(sp, ch) for sp, t in lines for ch in chunks(t)]
    return b"".join(await asyncio.gather(*jobs))


# ---------------------------------------------------------------- Feed
def esc(s):
    return html.escape(str(s), quote=True)


def write_feed(cfg, state, base):
    items = []
    for ep in state["episodes"]:
        desc = ep["description"].replace("]]>", "]]&gt;")  # Shownotes als HTML in CDATA
        items.append(f"""<item><title>{esc(ep['title'])}</title><description><![CDATA[{desc}]]></description>
<pubDate>{ep['pub']}</pubDate><guid isPermaLink="false">{ep['guid']}</guid>
<enclosure url="{base}/episodes/{ep['file']}" length="{ep['size']}" type="audio/mpeg"/>
<itunes:duration>{ep['duration']}</itunes:duration></item>""")
    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>
<title>{esc(cfg['podcast_title'])}</title><link>{base}/</link><language>de</language>
<description>Täglich automatisch erzeugtes Technik-Briefing</description>
<itunes:author>KI-Redaktion</itunes:author><itunes:explicit>false</itunes:explicit>
{chr(10).join(items)}
</channel></rss>"""
    (SITE / "feed.xml").write_text(feed, encoding="utf-8")
    (SITE / "index.html").write_text(
        f'<meta charset="utf-8"><title>{esc(cfg["podcast_title"])}</title>'
        f'<h1>{esc(cfg["podcast_title"])}</h1><p>Podcast-Feed: <a href="{base}/feed.xml">{base}/feed.xml</a></p>',
        encoding="utf-8")
    (SITE / ".nojekyll").write_text("")


# ---------------------------------------------------------------- main
def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    base = os.environ.get("SITE_URL", "http://localhost:8000").rstrip("/")
    (SITE / "episodes").mkdir(parents=True, exist_ok=True)
    sp = SITE / "state.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {"seen": [], "episodes": []}

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    MODELS["list"] = cfg.get("models") or [cfg["model"]]
    try:
        names = [m.name.replace("models/", "") for m in client.models.list()
                 if "generateContent" in (getattr(m, "supported_actions", None) or [])]
        print("Für diesen Key verfügbare Modelle:", ", ".join(n for n in names if "gemini" in n))
    except Exception as ex:
        print("Modellliste nicht abrufbar:", str(ex)[:100])
    day = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}"
    usage = state.get("llm_usage", {})
    budget_total = cfg.get("daily_llm_budget", 60)
    BUDGET["left"] = budget_total - (usage.get("count", 0) if usage.get("date") == day else 0)
    print(f"LLM-Budget heute: noch {BUDGET['left']} von {budget_total} Anfragen")
    seen = set(state["seen"])
    cands = [c for c in fetch_candidates(cfg) if c["id"] not in seen]
    if not cands:
        print("Nichts Neues."); return
    cands = prefilter(cfg, cands)
    cands.sort(key=lambda c: c["arxiv"])  # News/Journals zuerst, damit sie beim Kürzen nicht wegfallen (stabil)
    cands = cands[:cfg.get("max_candidates", 350)]
    if not cands:
        print("Nach Vorfilter nichts übrig."); return

    chosen, quick = rank(client, cfg, cands)
    print(f"Ausgewählt ({len(chosen)} Hauptthemen, {len(quick)} Kurzmeldungen, Modus: {cfg.get('content_mode')}):")
    for c, s, a in chosen:
        print(f" - [{s}|{a}] {c['title']}")
    for c in quick:
        print(f" . (kurz) {c['title']}")

    # Segmente parallel erzeugen (segment_workers aus config.yaml, Standard: 1)
    workers = max(1, cfg.get("segment_workers", 1))

    def _gen_seg(args):
        k, c = args
        try:
            seg = segment(client, cfg, c, k, len(chosen))
            return k, c, seg, None
        except RuntimeError as ex:
            return k, c, None, str(ex)

    body_parts, done_parts = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs = {pool.submit(_gen_seg, (k, c)): k for k, (c, _, _) in enumerate(chosen, 1)}
        for fut in as_completed(fs):
            k, c, seg, err = fut.result()
            if err:
                print(f"Thema {k} fehlgeschlagen: {err} -> übersprungen")
            elif not seg:
                print(f"Thema {k}: leeres Skript, übersprungen")
            else:
                print(f"Segment {k}: {sum(len(t.split()) for _, t in seg)} Wörter")
                body_parts[k] = seg
                done_parts[k] = c

    body, done = [], []
    for k in sorted(body_parts):
        body += body_parts[k]
        done.append(done_parts[k])

    if not done:
        state["llm_usage"] = {"date": day, "count": budget_total - BUDGET["left"]}
        sp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit("Keine Segmente erzeugt")

    # Kurzmeldungen: nicht besprochene Themen (je ca. 30 s), außerdem in den Shownotes
    qs = quick_summaries(client, cfg, quick)
    qspoken = [q for q in qs if q[1]]
    in_audio = cfg.get("quick_in_audio", True) and bool(qspoken)

    today = dt.datetime.now(dt.timezone.utc)
    intro, outro = intro_outro(client, cfg, [c["title"] for c in done], today, len(qspoken) if in_audio else 0)
    lines = list(intro) + body + (quick_lines(qspoken) if in_audio else []) + list(outro)

    audio = asyncio.run(synth(lines, cfg))
    fname = f"{today:%Y-%m-%d}.mp3"
    (SITE / "episodes" / fname).write_bytes(audio)
    secs = int(MP3(SITE / "episodes" / fname).info.length)

    ep = {"file": fname, "guid": f"{today:%Y-%m-%d}", "size": len(audio), "duration": secs,
          "pub": format_datetime(today),
          "title": f"{today.day}. {MONTHS[today.month-1]}: " + ", ".join(c["title"][:40] for c in done[:2]) + f" u.a. ({len(done)} Themen)",
          "description": build_notes(done, qs)}
    state["episodes"] = [e for e in state["episodes"] if e["file"] != fname]
    state["episodes"].insert(0, ep)
    for old in state["episodes"][cfg["keep_episodes"]:]:
        (SITE / "episodes" / old["file"]).unlink(missing_ok=True)
    state["episodes"] = state["episodes"][:cfg["keep_episodes"]]
    state["seen"] = (state["seen"] + [c["id"] for c in done] + [q[0]["id"] for q in qs])[-3000:]
    state["llm_usage"] = {"date": day, "count": budget_total - BUDGET["left"]}
    print(f"LLM-Anfragen heute insgesamt: {state['llm_usage']['count']}")
    sp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    write_feed(cfg, state, base)
    print(f"Fertig: {fname}, {secs//60} min {secs%60} s")


if __name__ == "__main__":
    main()