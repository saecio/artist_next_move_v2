# Decisions index

The decision log is split by theme; entries are never renamed, so an entry is cited by its exact title.

| File | Holds | Size (KB) |
|---|---|---|
| decisions_core.md | Methodology: principles, data and Works table, snapshots, Timing target, backtesting, Timing models and holdout, Style target and kill gate, Style models and holdout, global style forecast, audits, EDA and hypotheses, limitations and future work | 203.3 |
| decisions_demo.md | Build record of the Streamlit app; block 2 touches the predictions it shows | 148.4 |
| decisions_pending.md | Open items | 0.6 |
| decisions_archive.md | Reversed or replaced decisions (CUT) | 24.9 |

Rules:
- Read an entry with grep -n on its title and sed -n on the range, never the whole file.
- Older mentions of "docs/decisions.md" in code and notebooks, such as docs/decisions.md, entry "<title>", mean: look the title up in decisions_core.md or decisions_demo.md.

## decisions_core.md (line: heading)

3: ## DECIDED
5: ### 1. Principles and project setup (6 entries)
14: ### 2. Data and the Works table (41 entries)
58: ### 3. Snapshots and the Timing target (8 entries)
69: ### 4. Backtesting design (9 entries)
81: ### 5. Timing models, tuning and holdout (13 entries)
97: ### 6. Style target, baselines and kill gate (21 entries)
121: ### 7. Style models, tuning and holdout (16 entries)
140: ### 8. Global style forecast (9 entries)
152: ### 9. Audits and checks (4 entries)
157: #### Leakage in model selection: label look-ahead in development fold 5 (17-10-2026)
160: #### Duplicate audit: sensitivity analysis (17-10-2026)
168: #### Duplicate audit: sensitivity result (17-10-2026)
179: #### Style forecast: duplicate audit design (17-10-2026)
187: #### Style forecast: duplicate audit result (17-10-2026)
200: ### 10. EDA and hypotheses (12 entries)
204: #### EDA
208: #### Hypothesis Testing
242: ### 11. Limitations and future work (20 entries)
244: #### Future improvements
249: #### Limitations
267: ##### Known limitation: catalogue date (17-10-2026)

## decisions_demo.md (line: heading)

5: ## 1. Artist selection and pools
  7: ### Demo artist selection: eligibility rule (17-10-2026)
  20: ### Demo artist selection: cap per genre (17-10-2026)
  25: ### Demo artist selection: manual exclusions (17-10-2026)
  31: ### Demo artist selection: data-quality criterion (17-10-2026)
  41: ### Demo artist selection: final list (17-10-2026)
  55: ### Demo pool B: candidates and coverage rule (19-10-2026)
  65: ### Demo pool B: final list (19-10-2026)
  73: ### Demo pool B: WHEN assets (19-10-2026)
  82: ### Demo pool B: WHAT assets (19-10-2026)
  90: ### Demo: pool switch in tab 6 (19-10-2026)

98: ## 2. Live scoring assets
  100: ### Demo: re-recordings (17-10-2026)
  105: ### Demo: WHEN regeneration control (17-10-2026)
  109: ### Demo: timeline rule per artist and year, and baseline label (17-10-2026)
  113: ### Demo: live scoring with the final models (17-10-2026)
  122: ### Demo: live scoring assets, results (17-10-2026)
  129: ### Demo: live scoring assets for the style forecast, results (17-10-2026)
  138: ### Demo: works file of the demo artists (17-10-2026)
  145: ### Demo: WHEN scores file (17-10-2026)
  151: ### Demo: WHAT live assets, 2018 to the last data year (19-10-2026)
  160: ### Demo: WHAT labels after the last holdout year (19-10-2026)
  165: ### Demo: live style forecast for 2023 and later years (19-10-2026)
  174: ### Demo: WHEN scores for artist-years without a Work (20-10-2026)
  185: ### WHAT year-end rows (20-10-2026)
  195: ### Counting rule by reality year and the 2027 forecast (20-10-2026)

205: ## 3. Tab 6 (Time Machine)
  207: ### Demo: tab 6 flow and scoring (19-10-2026)
  219: ### Demo: route table for choosing a disclosed route (19-10-2026)
  229: ### Demo: tab 6 mode 2, several artists in one year (19-10-2026)
  249: ### Demo: tab 6 mode 1 layout: compact bar, windowed timeline, two-column panel (20-10-2026)
  259: ### Tab 6 panel wording, independent Timing and Style questions, new styles in the WHAT reality (20-10-2026)
  273: ### Demo: release title of the reality cells (20-10-2026)

281: ## 4. Tab 7 (style forecast)
  283: ### Demo: tab 7 shows the Random Forest (19-10-2026)

292: ## 5. Static tabs and story
  294: ### Static tabs of the demo (20-10-2026)
  300: ### Data chapters of the demo (21-10-2026)
  314: ### Method and closing slides (21-10-2026)
  326: ### Results depth views and calibration texts (22-10-2026)
  334: ### Tab 1 story strip (22-10-2026)
  340: ### Tab 4 inputs view and tab 2 cleaning view (22-10-2026)
  352: ### Trust view, result column, tab 7 subtitle and What works (22-10-2026)

360: ## 6. Text and wording rules
  362: ### Text rules for the audience (20-10-2026)
  367: ### Text and coherence pass (22-10-2026)
  373: ### Review fixes (22-10-2026)
  386: ### Tooltips, tab 5 chart layout and tab 6 disclaimer (22-10-2026)
  392: ### Wording fixes: tab 1 card 2 and tab 5 Style baseline line (23-10-2026)
  398: ### Wording: Release counts and tab 7 status line (23-10-2026)
  404: ### Wording: fourth item in tab 8 What comes next (24-10-2026)
  410: ### Wording: tab 4 models tooltip (24-10-2026)

416: ## 7. Design system and passes
  418: ### Design system of the demo (20-10-2026)
  424: ### Design pass 2: header, schemas, dot chart (20-10-2026)
  430: ### Design pass 3: story slides (20-10-2026)
  436: ### Design pass 8: alignment, gap wording, calibration wording and tab 1 hook with photos (22-10-2026)
  442: ### Design pass 10: typography and signature (22-10-2026)
  448: ### Design pass 11 (editorial identity and responsive) (22-10-2026)

## decisions_pending.md and decisions_archive.md

decisions_pending.md 1: ## PENDING
decisions_archive.md 3: ## CUT
