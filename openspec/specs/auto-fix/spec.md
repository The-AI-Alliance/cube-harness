# Auto-fix provenance & PI fix methodology

Authoritative spec for fixes produced by the **PrincipalInvestigator (PI)**
autonomous loop. Goal: let a reviewer trust most patches *without* a deep
contextual review, and surface design rot instead of accreting band-aids.

The PI runs without a feedback channel to other users. So a fix cannot be
validated by asking "does this help other cubes?" — it must instead carry
its own generalization proof, an adversarial audit of that proof, and a
durable, context-stamped provenance record so a future agent (or a
different-context PI run) can see what was patched, why, and where it may
not generalize.

## 1. Depth taxonomy (classify every fix)

| Class | Meaning | Policy |
|---|---|---|
| **L0** local-correct | restores an invariant/symmetry that already exists elsewhere; no contract change | trust-by-default |
| **L1** layer fix | correct but touches a shared layer/call site | trust if fix-audit clears blast-radius |
| **L2** symptom-of-design | the bug is a symptom; the patch is a band-aid | ship temp PR **and** open a kept-open `design-debt` issue + openspec stub |
| **L3** PI/eval defect | the defect is in the investigator/methodology itself, not the cube | ship + flag; do not patch the cube |

**Nothing blocks the loop.** L2/L3 still ship a temporary PR. The human's
queue is the `design-debt` backlog, not individual patches.

## 2. The Fix Dossier (every fix PR carries its own proof)

Template: `.claude/skills/principal-investigator/templates/fix_dossier.md`.
It is the PR body. It must state:

- **Invariant** — the violated rule in one sentence (not "task X failed").
- **Blast radius** — every call site / consumer, grep run **on the PR's
  target branch** (not the local checkout — feature branches lie).
- **Adversarial "opposite user"** — construct the downstream context this
  fix could *hurt*; rule it out or escalate to L2.
- **Registry lookup** — existing `auto-fix` footnotes in/near the area and
  their context fingerprints (this is the loop's substitute for real
  cross-user feedback).
- **Class** (L0–L3) and, for L2/L3, the design issue / openspec link.
- **Test** — a regression test asserting the *invariant*, not the
  reproduction.

## 3. The fix-audit

An independent agent gets the diff + Dossier and is instructed to **break
the generalization claims** (find an unchecked call site, construct the
hurt downstream user, argue the fix is shallow). The reviewer reads the
audit verdict, not the raw diff. Recipe:
`analyze/investigator/use_cases/fix_audit/` (mirrors the audit pass).

## 4. In-code provenance

### Marker (one line each, minimal pollution)

```python
# auto-fix(345)↓
<changed code>
# /auto-fix(345)
```

`345` is a **GitHub issue number** (see §5) — unique, monotonic, durable,
holds the long-form context out of the code.

### Footnote (module bottom — durable anchor)

```python
# === auto-fix notes ===
# auto-fix-note(345) {class=L1 issue=345 hash=ab12cd34 ctx=colima/arm64/gpt-5.4-mini/cube@285e0cc}
#   symptoms:  <what was observed + the triggering context (see note)>
#   invariant: <the rule that was being violated>
#   why:       <why this fix, why this layer>
#   tested:    <the regression test / how verified>
```

Two sections per stanza: a **machine line** in `{...}` (lint may rewrite
`hash`; never hand-edit) and **prose** (PI/human authored; never
auto-touched). `ctx=` is the deterministic machine-matchable minimum
(infra/arch/model/cube-commit). `symptoms` additionally records the
**triggering context in prose** — the task/benchmark that surfaced the
bug plus the env specifics (OS, infra backend, model, input shape, …)
the agent **judges relevant to *this* bug**. Selection is contextual,
not a fixed checklist: a docker-socket bug names the backend/OS; a
parsing bug names the task and input shape; a model-behaviour bug names
the model and task. Enough for a different-context run to reason about
whether the fix generalises — not an exhaustive environment dump.

## 5. Lifecycle (GitHub issue as the ID source)

Issue-first resolves the ID-ordering problem (you need the ID to write the
marker, before the PR exists):

1. Open issue → get `N`. Body = full Dossier + context fingerprint.
   Label `auto-fix`.
2. Write `auto-fix(N)` markers + footnote; open the fix PR referencing `N`.
3. **L0/L1:** close issue `N` on merge (pure archive).
   **L2/L3:** keep issue `N` open, add label `design-debt`, link the
   openspec stub. This is the backlog the human reviews.
4. **Consolidation:** when a refactor promotes band-aids to permanent
   design, it deletes the marked regions + footnotes and the refactor
   PR/openspec cites `supersedes auto-fix N, M, …`; the issues become the
   audit trail of what the refactor subsumed.

Degraded mode: if `gh` is unavailable at fix time, use
`auto-fix(LOCAL-<uuid>)`; a reconciliation pass upgrades it to a real `N`.

## 6. Rot detection — two tiers

**Tier 1 — deterministic lint (`scripts/auto_fix_lint.py`, CI):**

- markers balanced; no nesting/boundary mismatch; close-after-open
- every inline `N` has a footnote stanza and vice-versa (no orphans)
- `N` unique; footnote machine-line schema valid; issue `N` exists with
  the right open/closed + label state
- **content-hash drift**: normalized hash of the marked region vs the
  footnote's `hash=`; mismatch ⇒ protected code moved.

**Tier 2 — semantic (fix-audit / `/cube-review`, advisory):** is the
changed block still correct / still needed / now subsumed? Lint flags;
an agent judges. Lint never rules on semantics.

### Acknowledgement, not prevention

Hash drift is **not** a hard CI fail — that makes auto-fix blocks
radioactive and people delete markers to go green. The failure condition
is *unacknowledged* drift. Resolve by either: (a) editor updates the
footnote + re-hash (acknowledges they preserved the fix), (b) route to
fix-audit, or (c) remove the marker **with a closing note on issue `N`**.
CI flags; a human/agent must acknowledge.

## 7. Accretion → refactor

Density is now exact: `grep -c 'auto-fix(' <area>`. When an area crosses
the band-aid threshold (≥3), the PI **must** emit an openspec refactor
proposal instead of patch N+1. Patch accretion is a design signal, not a
steady state.
