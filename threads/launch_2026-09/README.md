# SCT launch package (September 2026)

Posting-ready copy for the Substack article and the X threads. The source is Kolmogorov's 2026-09-18 drafts (mail `m_a83edecf1c73422b78001485`) and threads 01, 02, 04, and 05 in `threads/`. Claude edited this copy on 2026-09-19 with the `blader/humanizer` v3.0.0 rules (MIT, github.com/blader/humanizer) and checked every number against the result files.

Kolmogorov: this folder is in your reach at `/reach/ai_research/topics/structure_of_clear_thinking/threads/launch_2026-09/`. Use these files instead of the versions in your 2026-09-18 mail.

## Posting order

1. `substack_post.md`: publish first, so the threads can link to it.
2. `kickoff_tweet.md`: pick option A or B and pin it.
3. `thread_1_reachability.md`: day 1.
4. `thread_2_receipts.md`: day 2.
5. `thread_3_map_not_in_attention.md`: day 3.
6. `thread_4_replication.md`: day 4. Unpin the kickoff tweet after this thread.

Each post is 280 characters or less (URLs counted as 23, the thread emoji as 2). Each image has alt text under its post. Reply posts go under the post named in their heading.

## Fact corrections to the 2026-09-18 drafts

1. NeSy 2026. The drafts said the abstract and paper deadlines were met and notifications arrived July 8. The OpenReview upload never happened, and the paper is now a self-published preprint (`.swarm/state.md`, SCT-007 closed 2026-09-18). The kickoff tweet and the Substack timeline now say this.
2. Reachability masking. The draft said masking "didn't move the needle". Masking raised valid sequences from 48.1% to 61.5% for the well-calibrated model and from 4.2% to 4.3% for the mis-calibrated model (`Percepta_Transformer_VM/experiment_results.md`).
3. H¹ contradiction test. The draft placed the 5 to 58 result on FB15K-237 and WN18RR. The report gives it for a separate clean subgraph (`results/week4_evaluation_report.md`, conflict detection table).
4. "Hallucinations architecturally impossible." The guarantee covers rule compliance under the rulebook in force. The copy now says that.
5. Epistemic-status gate. The code and tests exist. Nobody has tested the gate on a trained model, so the copy now says so.
6. "Everything reproduces on a laptop." Threads 1 and 2 run on a laptop. Threads 3 and 4 used a cloud L4 GPU.
7. Thread 4 said "last week" and "threads 1 to 4". The experiment ran in July, and the SCT thread 03 (teaching the rulebook) is not in this launch, so the copy now uses the launch order.

## Open items for Mike

1. The competitor section in `substack_post.md` is marked OPTIONAL. Claude could not verify the other lab, its founders, or its launch date. Verify these details or delete the section. Kolmogorov's own rule applies to the threads: do not name the other lab unless someone else raises it first.
2. `papers/nesy_submission/main.pdf` is not in git (`*.pdf` is in `.gitignore`), so the README link to the preprint does not work on GitHub. Before the Substack post goes out, either commit the PDF with `git add -f` or remove the preprint link.
3. Kolmogorov could not install the Humanizer skill. Every `openclaw skills` call and the `skill_workshop apply` call returned "Plugin approval required (gateway unavailable)". The proposal `humanizer-20260919-3ff5d1f6de` is pending. `openclaw status` shows a pending device approval (`fe440e4b-b1df-4d5f-8f68-38a928142d8e`). Only Mike can approve these.
4. SCT thread 03 (teaching the rulebook) and blog posts 06 to 08 are not in this launch. They can follow as a second round.
