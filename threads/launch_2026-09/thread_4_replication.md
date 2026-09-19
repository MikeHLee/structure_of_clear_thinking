# Thread 4 (day 4): The result that died in replication

Source draft: `threads/05_placement_replication/thread.md`
Figures: `threads/05_placement_replication/figures/`

## Post 1  [attach: fig1_effect_dissolves.png]

In July, one of our experiments said a model generalizes rules about twice as well when fine-tuning is limited to attention weights. It had a plausible mechanism and p≈0.07.

We committed to a replication before posting it. The effect did not replicate. Here are the results. 🧵

Alt text: Chart titled "One seed looked exciting. Three seeds say: no effect." Four conditions (attention routing q/k, attention read-out v/o, MLPs only, all weights), each with three seed dots and a pooled bar: 19.2%, 20.0%, 15.0%, and 16.7% rule-following on never-trained regions. The spread within each condition is larger than the differences between conditions; chi-squared p = 0.46.

## Post 2

The question: when a model learns a rulebook, does it matter which weights the gradient can change?

4 conditions: attention routing (q/k), attention read-out (v/o), MLPs only, all weights. Same corpus, same batch order, scored on ontology regions the model never saw.

## Post 3

On seed 42, read-out-only tuning reached 96% on trained regions and about twice the transfer of MLP tuning to unseen regions, with a quarter of the parameters. It matched the unused-knowledge head from the last thread.

Our commit message said "directional, needs seeds."

## Post 4

Then we ran seeds 43 and 44 with nothing else changed.

On seed 43, read-out came last. On seed 44, all-weights, which was last on seed 42, came second.

Pooled (n=240 per condition): 19.2%, 20.0%, 15.0%, 16.7%. χ² p=0.46. The seed-42 ordering was noise.

## Post 5

At this scale, we have no reliable evidence that LoRA placement changes how well rule-following transfers to new regions.

Please do not cite the seed-42 numbers as a finding. We are posting this partly so that nobody, us included, brings them back later.

## Post 6  [attach: fig2_generalization_wall.png]

One result held in all 12 runs (3 seeds × 4 conditions): trained regions reached 85 to 92% rule-following, and never-seen regions stayed at 15 to 20%. No choice of weights or rank changed that.

Fine-tuning teaches the regions it sees and not the rulebook as a whole.

Alt text: Grouped bar chart titled "What replicates in every run: the generalization wall." In all four conditions, rule-following is 85 to 92% on trained regions and 15 to 20% on never-seen regions. Twelve runs across three seeds.

## Post 7

Prior work already found that attention-only tuning is not special for in-domain learning: "LoRA Learns Less and Forgets Less" (arXiv:2405.09673) and Thinking Machines' "LoRA Without Regret."

We measured a different axis, transfer to new regions, and also found no effect.

## Post 8

Why the first result was easy to believe: one seed, p≈0.07, a plausible mechanism, and a result we wanted. Those are the conditions under which a result is most likely to be wrong.

The replication cost $4 of GPU time and one day.

## Post 9

For the project, this means you cannot pick which weights to fine-tune and get rule-following that generalizes. The gap between trained and new regions was there in every configuration.

So the guarantee comes from per-step enforcement and a receipt (threads 1, 2).

## Post 10

All three seeds, all four conditions, and the replication commitment in the commit history:

github.com/MikeHLee/structure_of_clear_thinking

The earlier threads are linked below.
