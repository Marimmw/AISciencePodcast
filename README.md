# Täglicher KI-Podcast (kostenlos)

## Einmalige Einrichtung (ca. 20-30 Minuten)
1. **Gemini-API-Key** holen (kostenlos): https://aistudio.google.com -> "Get API key".
2. **GitHub-Account** + neues **öffentliches** Repository anlegen (nötig für kostenloses GitHub Pages).
3. Alle Dateien dieses Ordners hochladen (inkl. des versteckten Ordners `.github/workflows/`).
4. Repo -> Settings -> Secrets and variables -> Actions -> **New repository secret**:
   Name `GEMINI_API_KEY`, Wert = dein Key.
5. Repo -> Actions -> "daily-podcast" -> **Run workflow** (erster Testlauf, dauert ca. 10 min).
6. Danach: Settings -> Pages -> Source "Deploy from a branch" -> Branch `gh-pages` / root -> Save.
7. Feed-URL: `https://DEIN-NAME.github.io/REPO-NAME/feed.xml`
   In AntennaPod / Pocket Casts / Apple Podcasts ("Sendung per URL hinzufügen") abonnieren.

Ab jetzt entsteht jeden Morgen automatisch eine neue Folge.

## Anpassen
Alles in `config.yaml`: Interessen, Länge, Quellen, Stimmen.
Lokal testen: `pip install -r requirements.txt`, dann
`GEMINI_API_KEY=... SITE_URL=http://localhost:8000 python podcast.py`.

## Hinweise
- GitHub pausiert Zeitpläne nach 60 Tagen ohne Repo-Aktivität: dann einmal "Enable workflow" klicken.
- Die Feed-URL ist öffentlich (nicht gelistet, aber erreichbar). Nur eigene Zusammenfassungen, keine Volltexte.
- Free-Tier-Limits von Gemini ändern sich; das Skript wartet bei Fehlern automatisch.
- edge-tts nutzt die inoffizielle Edge-Vorlesestimme von Microsoft. Falls sie mal ausfällt: Stimmen in config ändern oder auf Piper wechseln.

## Neu: Modi und Länge
- `content_mode: "fulltext"` in `config.yaml` -> Papers werden als PDF gelesen (Methode, Ergebnisse, Grenzen), Artikel als Volltext.
  Der Lauf dauert dann länger (ca. 20-40 min), das ist ok.
- Länge: `min_minutes` bis `max_minutes` (z.B. 20-60). Wie viele Themen es wirklich werden, entscheidet die Bewertung (`min_score`).
- Ein 60-Minuten-Lauf braucht ca. 18 Gemini-Anfragen pro Tag (weit unter dem Free-Tier-Limit).
- 60 min MP3 = ca. 20 MB. Bei `keep_episodes: 14` sind das ~300 MB auf GitHub Pages (Limit: 1 GB). Bei Bedarf kleiner stellen.

## Checkliste: garantiert im Kostenlosen bleiben
1. Google Cloud Console -> Billing: dem Gemini-Projekt darf KEIN Rechnungskonto zugeordnet sein.
2. API-Key nur als GitHub-Secret speichern, nie in Dateien/Chat/Repo. Key in AI Studio/Cloud Console auf "Generative Language API" beschränken.
3. GitHub -> Settings -> Billing and licensing -> Budgets: Limit auf 0 $ mit "Stop usage" setzen.
4. Repo öffentlich lassen (Standard-Runner und Pages sind dann kostenlos).
5. `daily_llm_budget` in config.yaml begrenzt die Gemini-Anfragen pro Tag hart.
6. Einmal pro Woche: Actions -> Läufe grün? Pages-Größe unter 1 GB (`keep_episodes` senken).
