# Media assets

Screenshots and recordings referenced from the project README and docs. Keep
these lightweight (PNG ≤ ~300 KB, or an SVG / optimized GIF).

## Wanted

| File | Used by | What to capture |
| --- | --- | --- |
| `runaway-cost-demo.png` | `README.md` (Try the No-Key Demo) | Terminal output of `uv run python examples/runaway_cost_prevention.py` — show the loop running and the final `RunResult` with `success: false` and `terminated_reason` set to the budget cap. This is the headline "it works" shot. |
| `architecture.svg` *(optional)* | `README.md` (Why This Exists), portfolio | A clean vector version of the flow diagram (your agent / LangGraph / OpenAI Agents SDK → GuardLoop → BudgetController / CircuitBreakerRegistry / VerifierChain / OpenTelemetry / RunContext). The Mermaid block in the README already renders on GitHub; this is only worth it if you want a polished standalone image for the portfolio page. |
| `verifier-retry-demo.png` *(optional)* | `README.md` (Try the No-Key Demo) | Terminal output of `uv run python examples/verifier_retry_loop.py` showing the bad → feedback → corrected → `verification_passed: true` sequence. |
| `social-card.png` *(optional)* | GitHub repo "Social preview" setting (Settings → General) | 1280×640. Title "GuardLoop", one-line tagline, the four pillar words. Not embedded in the README — set it under repo settings. |

Once you add `runaway-cost-demo.png` the placeholder in `README.md` resolves; the
rest are optional polish.
