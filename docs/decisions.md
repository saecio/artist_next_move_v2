# Decisions index

Decision log split by theme. Entries are moved verbatim from the former single file.

| File | Holds | Size |
|---|---|---|
| decisions_core.md | Data, snapshots, WHEN, WHAT, duplicate audits, leakage, models, holdout, style forecast, EDA, hypotheses, limitations | 211 KB |
| decisions_demo.md | Demo app: artist selection, pool B, tabs, text rules, design, counting rule, year-end rows | 108 KB |
| decisions_pending.md | Open items (PENDING) | 2 KB |
| decisions_archive.md | History of reversed decisions (CUT), not needed for normal work | 24 KB |

Rule: cite an entry by its exact title. Read an entry with `grep -n` on the title and `sed -n` on the range, never the whole file.

Entry titles per file (line number in that file):

decisions_core.md:
3: DECIDED
140: Leakage in model selection: label look-ahead in development fold 5 (17-10-2026)
143: Duplicate audit: sensitivity analysis (17-10-2026)
151: Duplicate audit: sensitivity result (17-10-2026)
162: EDA
166: Hypothesis Testing
200: Future improvements
204: Limitations
222: Known limitation: catalogue date (17-10-2026)
231: Style forecast: duplicate audit design (17-10-2026)
239: Style forecast: duplicate audit result (17-10-2026)

decisions_demo.md:
1: Demo artist selection: eligibility rule (17-10-2026)
14: Demo artist selection: cap per genre (17-10-2026)
19: Demo artist selection: manual exclusions (17-10-2026)
25: Demo artist selection: data-quality criterion (17-10-2026)
35: Demo artist selection: final list (17-10-2026)
49: Demo: re-recordings (17-10-2026)
54: Demo: WHEN regeneration control (17-10-2026)
58: Demo: timeline rule per artist and year, and baseline label (17-10-2026)
62: Demo: live scoring with the final models (17-10-2026)
71: Demo: live scoring assets, results (17-10-2026)
78: Demo: live scoring assets for the style forecast, results (17-10-2026)
87: Demo: works file of the demo artists (17-10-2026)
94: Demo: tab 6 flow and scoring (19-10-2026)
101: Demo: tab 7 shows the Random Forest (19-10-2026)
110: Demo: WHAT live assets, 2018 to the last data year (19-10-2026)
119: Demo: WHAT labels after the last holdout year (19-10-2026)
124: Demo pool B: candidates and coverage rule (19-10-2026)
134: Demo pool B: final list (19-10-2026)
142: Demo pool B: WHEN assets (19-10-2026)
151: Demo pool B: WHAT assets (19-10-2026)
159: Demo: pool switch in tab 6 (19-10-2026)
167: Demo: route table for choosing a disclosed route (19-10-2026)
174: Demo: tab 6 mode 2, several artists in one year (19-10-2026)
186: Demo: tab 6 mode 1 layout: compact bar, windowed timeline, two-column panel (20-10-2026)
193: Tab 6 panel wording, independent Timing and Style questions, new styles in the WHAT reality (20-10-2026)
202: Demo: release title of the reality cells (20-10-2026)
208: Demo: live style forecast for 2023 and later years (19-10-2026)
217: Demo: WHEN scores for artist-years without a Work (20-10-2026)
228: WHAT year-end rows (20-10-2026)
238: Counting rule by reality year and the 2027 forecast (20-10-2026)
248: Text rules for the audience (20-10-2026)
253: Static tabs of the demo (20-10-2026)
259: Design system of the demo (20-10-2026)
264: Design pass 2: header, schemas, dot chart (20-10-2026)
269: Design pass 3: story slides (20-10-2026)
274: Demo: WHEN scores file (17-10-2026)

decisions_pending.md:
1: PENDING

decisions_archive.md:
3: CUT
