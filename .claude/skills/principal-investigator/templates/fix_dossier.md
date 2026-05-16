# Fix Dossier — auto-fix(<N>)

> This is the fix PR body. Spec: `openspec/specs/auto-fix/spec.md`. Fill every field;
> the fix-audit will try to break the claims below.

**Class**: <L0 | L1 | L2 | L3>
**Issue**: #<N>  ·  **Design-debt issue / openspec** (L2/L3 only): <link>
**Context fingerprint**: <infra>/<arch>/<model>/<cube@commit>

## Invariant violated
<One sentence: the rule the code must uphold. NOT "task X failed".>

## Root cause vs symptom
<Why this is the cause, not the stack trace. For L2/L3: name the design
defect and why a band-aid is shipping anyway.>

## Blast radius
<Every call site / consumer. Paste the grep, run on the PR's TARGET
branch (`git grep … origin/dev`), not the local checkout.>

## Adversarial: the opposite user
<Construct the downstream context (cube / infra / model) this fix could
HURT. Rule it out with evidence, or escalate the class.>

## Registry lookup
<Existing `auto-fix` footnotes in/near this area + their `ctx=`
fingerprints. Is this the Nth patch here? If ≥3 → refactor proposal, not
patch N+1.>

## Fix & layer
<The change, and why this is the correct layer (cube / cube-standard /
scaffold / eval). If patched in a consumer but the defect is in the
contract → wrong layer → escalate.>

## Test
<Regression test asserting the INVARIANT, not the reproduction. Path +
one line on what it pins.>
