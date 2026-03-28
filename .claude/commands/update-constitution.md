# Update Constitution

This command updates the cube-harness Constitution and propagates changes to all dependent files.

## Instructions

When the user wants to update the constitution rules:

1. **Read the current constitution** from `.claude/rules/constitution.md`

2. **Apply the requested changes** to the constitution document

3. **Update the detailed rules** in `.claude/rules/review-rules.md` if:
   - New directives are added (add corresponding rule with ID, severity, and examples)
   - Existing directives are modified (update the detailed explanation)
   - Directives are removed (remove from the detailed rules)

4. **Update the PR template checklist** in `.github/PULL_REQUEST_TEMPLATE/general_code_pr.md`:
   - Update the "Constitution Compliance" section to reflect current rules
   - Only include the most important/common rules in the checklist

5. **Verify consistency** across all files:
   - Rule IDs match (TC-001, EX-001, etc.)
   - Severity levels are consistent
   - Descriptions align with constitution directives

## Files to Update

| File | What to Update |
|------|----------------|
| `.claude/rules/constitution.md` | **Source of truth** - The full constitution |
| `.claude/rules/review-rules.md` | Enforceable rules with severity levels and code examples |
| `.github/PULL_REQUEST_TEMPLATE/general_code_pr.md` | Checklist items |

## Example Usage

User: "Add a new directive under Pillar V for requiring docstrings on public functions"

Then update:
1. Add the directive to constitution.md under Pillar V: The Craft of Code
2. Add detailed rule CC-006 to review-rules.md with severity and examples
3. Optionally add to PR template checklist if it's a common check
