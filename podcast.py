#!/usr/bin/env python3
"""Täglicher KI-Podcast: Quellen holen -> ranken -> Skripte -> TTS -> RSS-Feed."""
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


def rank(client, cfg, cands):
    mpi = cfg["minutes_per_item"]
    n_min = max(1, round((cfg["min_minutes"] - 1) / mpi))
    n_max = max(n_min, round((cfg["max_minutes"] - 1) / mpi))
    listing = "\n".join(f"[{i}] ({c['source']}) {c['title']} — {c['text'][:350]}" for i, c in enumerate(cands))
    prompt = f"""Du bist Redakteur eines technischen Podcasts für einen ingenieurwissenschaftlich interessierten Hörer.

INTERESSEN:
{cfg['interests']}
UNINTERESSANT:
{cfg['avoid']}

Bewerte die Einträge und wähle bis zu {n_max} aus. Vergib einen score von 1-10
(10 = sensationell neu und genau passend; 7 = lohnt sich in einer Folge; darunter = überspringen).
Bevorzuge echte technische Neuheiten und überraschende Konzepte vor PR und Meinung, sorge für Themenmischung.
Lieber weniger Einträge als Füllmaterial.
Antworte NUR mit JSON: [{{"id": <Nummer>, "score": <1-10>, "why": "<ein Satz>"}}], beste zuerst.

LISTE:
{listing}"""
    data = json.loads(llm(client, cfg, prompt, json_mode=True))
    ranked, used = [], set()
    for d in data:
        i, s = d.get("id"), d.get("score", 0)
        if isinstance(i, int) and 0 <= i < len(cands) and i not in used:
            used.add(i); ranked.append((cands[i], s if isinstance(s, (int, float)) else 0))
    ranked.sort(key=lambda x: -x[1])
    chosen = [(c, s) for c, s in ranked if s >= cfg["min_score"]][:n_max]
    if len(chosen) < n_min:
        chosen = ranked[:n_min]
    return chosen


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


def intro_outro(client, cfg, titles, today):
    datum = f"{WEEKDAYS[today.weekday()]}, {today.day}. {MONTHS[today.month-1]}"
    p = f"""Schreibe Intro und Outro für den Podcast "{cfg['podcast_title']}" als Gespräch zwischen LENA und JONAS (Deutsch, locker).
Heute ist {datum}. Themen der Folge:
{chr(10).join('- ' + t for t in titles)}
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
        items.append(f"""<item><title>{esc(ep['title'])}</title><description>{esc(ep['description'])}</description>
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

    chosen = rank(client, cfg, cands[:300])
    print(f"Ausgewählt ({len(chosen)} Themen, Modus: {cfg.get('content_mode')}):")
    for c, s in chosen:
        print(f" - [{s}] {c['title']}")

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
        fs = {pool.submit(_gen_seg, (k, c)): k for k, (c, _) in enumerate(chosen, 1)}
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

    today = dt.datetime.now(dt.timezone.utc)
    intro, outro = intro_outro(client, cfg, [c["title"] for c in done], today)
    lines = list(intro) + body + list(outro)

    audio = asyncio.run(synth(lines, cfg))
    fname = f"{today:%Y-%m-%d}.mp3"
    (SITE / "episodes" / fname).write_bytes(audio)
    secs = int(MP3(SITE / "episodes" / fname).info.length)

    desc = "Themen:\n" + "\n".join(f"- {c['title']} ({c['source']})\n  {c['link']}" for c in done)
    ep = {"file": fname, "guid": f"{today:%Y-%m-%d}", "size": len(audio), "duration": secs,
          "pub": format_datetime(today),
          "title": f"{today.day}. {MONTHS[today.month-1]}: " + ", ".join(c["title"][:40] for c in done[:2]) + f" u.a. ({len(done)} Themen)",
          "description": desc}
    state["episodes"] = [e for e in state["episodes"] if e["file"] != fname]
    state["episodes"].insert(0, ep)
    for old in state["episodes"][cfg["keep_episodes"]:]:
        (SITE / "episodes" / old["file"]).unlink(missing_ok=True)
    state["episodes"] = state["episodes"][:cfg["keep_episodes"]]
    state["seen"] = (state["seen"] + [c["id"] for c in done])[-3000:]
    state["llm_usage"] = {"date": day, "count": budget_total - BUDGET["left"]}
    print(f"LLM-Anfragen heute insgesamt: {state['llm_usage']['count']}")
    sp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    write_feed(cfg, state, base)
    print(f"Fertig: {fname}, {secs//60} min {secs%60} s")


if __name__ == "__main__":
    main()
