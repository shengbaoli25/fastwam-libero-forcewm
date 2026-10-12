# ForceWM supervisor PID reuse recovery

The 2026-10-12 four-H200 run stopped at the last logged step 4440 after a previously tracked PID was recycled. The outer supervisor raised `Refusing reused process identity` and terminated training. No checkpoint had reached the configured first save at step 5000.

A process identity is `(PID, start_ticks)`, not PID alone. The tracker now archives an old identity when the same PID has a strictly later start time. It retains the original ancestry in the receipt and discovers a replacement independently from a currently verified parent or exclusive subreaper adoption. An unrelated replacement is never inherited or signalled. Signal and reap races use the same retirement rule; every actual signal still requires a pidfd and an exact identity check. Backward identities and an unreaped registered launcher changing identity remain errors. A transient terminal `/proc` row can resolve to an absent process or a valid later identity; persistent invalid metadata remains an error.

The restart validator may reuse the existing same-allocation four-rank GPU probe only if every model/training Python file and distributed configuration is unchanged. Only the outer supervisor files and the output directory may differ. Original receipts are retained. This is reuse of prior GPU evidence, not a new GPU test. Each run has separate output, origins, supervision, and hash-bound plan files.

Validation: 63 CPU/process cases pass, including 13 PID-reuse cases, previous terminal-state races, budget/config guards, and real four-session success/failure/deadline cleanup. Synthetic process identity injection is test-only; these tests do not establish model training, checkpoint saving, or restore success.

The original fixed deadline remains in force until the user explicitly approves a change. Restart from the complete official Base is required because no checkpoint exists. Recipe: four ranks, batch16 per rank, accumulation1, global64, max30000, save5000. All cached video features and model mathematics are unchanged.

The user subsequently approved “同意延长到50小时并重启”. The new `forcewm-restart50h-v4` plan applies only to allocation `ab5d6993-795b-4bf7-9084-e59f089e7788` and retains the original budget anchor. Its fixed deadline is 2026-10-14 02:02:25 Asia/Shanghai (Unix 1791914545). The previous 48-hour plan and failed run remain unchanged. Seven additional budget cases cover this approval and reject resetting the clock, another allocation, a missing approval, changed anchor/recipe, and insufficient remaining time.
