commit 76cfd97293289d1f1fc58fac8ac6c6822befeedf
Author: recursix <alex.lacoste.shmu@gmail.com>
Date:   Thu Apr 23 19:35:16 2026 -0400

    fix(workarena): shorten infeasible tool docstring; activate CLI arg parsing in recipe
    
    - tools.py: trim report_infeasible docstring (guidance on when to use it moved to hints)
    - workarena_l1_full.py: replace hardcoded main() call with sys.argv CLI parsing
      (debug, headless-off, hints, retry <path>, model selection flags)
    
    Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
    
    Signed-off-by: recursix <alex.lacoste.shmu@gmail.com>

diff --git a/cubes/workarena/src/workarena_cube/tools.py b/cubes/workarena/src/workarena_cube/tools.py
index b1dcd34..844c820 100644
--- a/cubes/workarena/src/workarena_cube/tools.py
+++ b/cubes/workarena/src/workarena_cube/tools.py
@@ -54,7 +54,7 @@ class WorkArenaInfeasibleTool(Tool):
 
     @tool_action
     def report_infeasible(self, explanation: str) -> str:
-        """Report that this task is genuinely impossible to complete. Use ONLY when the task is objectively infeasible: a required ServiceNow record does not exist, the instructions are contradictory, or a needed feature or field is absent. Do NOT use when the task is difficult or you are uncertain — always attempt the task first. Calling this immediately ends the episode.
+        """Report that this task is genuinely impossible to complete. Use ONLY when the task is objectively infeasible.
 
         Args:
             explanation: Brief explanation of why the task cannot be completed.
