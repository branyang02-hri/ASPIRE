# F1TENTH Clean Campaign

A new evaluation requires a new campaign ID and an empty path under
`outputs/f1tenth/aspire-campaigns/`. Do not delete or modify an old campaign.

Before launch:

1. Confirm the target campaign path does not exist or is empty.
2. Confirm no candidate, incumbent, skill, manifest, or history is copied from
   another campaign.
3. Use the tracked `initial_controller.py`.
4. Record the selected model, reasoning effort, and iteration budget at `init`.
5. Let `campaign.py init` establish fresh development and validation baselines.

If a campaign is interrupted, resume its current machine state. Do not call it
a fresh run and do not create replacement held-out episodes.
