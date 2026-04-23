commit 76cfd97293289d1f1fc58fac8ac6c6822befeedf
Author: recursix <alex.lacoste.shmu@gmail.com>
Date:   Thu Apr 23 19:35:16 2026 -0400

    fix(workarena): shorten infeasible tool docstring; activate CLI arg parsing in recipe
    
    - tools.py: trim report_infeasible docstring (guidance on when to use it moved to hints)
    - workarena_l1_full.py: replace hardcoded main() call with sys.argv CLI parsing
      (debug, headless-off, hints, retry <path>, model selection flags)
    
    Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
    
    Signed-off-by: recursix <alex.lacoste.shmu@gmail.com>

diff --git a/meta_agent/recipes/workarena_l1_full.py b/meta_agent/recipes/workarena_l1_full.py
index c554e23..8350a06 100644
--- a/meta_agent/recipes/workarena_l1_full.py
+++ b/meta_agent/recipes/workarena_l1_full.py
@@ -131,38 +131,29 @@ def main(
 
 
 if __name__ == "__main__":
+    args = sys.argv[1:]
+    args_set = set(args)
+    debug = "debug" in args_set
+    headless = not debug and "headless-off" not in args_set
+    use_hints = "hints" in args_set
+
+    retry_dir: Path | None = None
+    if "retry" in args_set:
+        retry_idx = args.index("retry")
+        if retry_idx + 1 < len(args):
+            retry_dir = Path(args[retry_idx + 1])
+        else:
+            print("ERROR: 'retry' flag requires a path argument", file=sys.stderr)
+            sys.exit(1)
+
+    selected = [k for k in MODEL_CONFIGS if k in args_set]
+    if not selected:
+        selected = ["gpt-5.4-mini"]
+
     main(
-        debug=False,
-        headless=True,
-        models=["gpt-5.4-mini"],
-        use_hints=False,
-        retry_dir=None,
+        debug=debug,
+        headless=headless,
+        models=selected,
+        use_hints=use_hints,
+        retry_dir=retry_dir,
     )
-
-# if __name__ == "__main__":
-#     args = sys.argv[1:]
-#     args_set = set(args)
-#     debug = "debug" in args_set
-#     headless = not debug and "headless-off" not in args_set
-#     use_hints = "hints" in args_set
-
-#     retry_dir: Path | None = None
-#     if "retry" in args_set:
-#         retry_idx = args.index("retry")
-#         if retry_idx + 1 < len(args):
-#             retry_dir = Path(args[retry_idx + 1])
-#         else:
-#             print("ERROR: 'retry' flag requires a path argument", file=sys.stderr)
-#             sys.exit(1)
-
-#     selected = [k for k in MODEL_CONFIGS if k in args_set]
-#     if not selected:
-#         selected = ["gpt-5.4-mini"]
-
-#     main(
-#         debug=debug,
-#         headless=headless,
-#         models=selected,
-#         use_hints=use_hints,
-#         retry_dir=retry_dir,
-#     )
