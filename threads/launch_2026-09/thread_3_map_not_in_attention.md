# Thread 3 (day 3): The map is not in the attention

Source draft: `threads/04_map_not_in_attention/thread.md`
Figures: `threads/04_map_not_in_attention/figures/`

## Post 1  [attach: fig1_soft_graph_exists.png]

We looked for a knowledge graph inside a language model's attention heads and found one. Some heads tell real ontology relations from fake ones at up to 0.90 AUC, on scrambled text, before any training.

Then we deleted those heads. Rule-following did not change. 🧵

Alt text: Histogram titled "Some attention heads already know the ontology, before any training." It shows the rule-detection AUC of each of Qwen2.5-1.5B's 336 attention heads on scrambled sequences. Most heads are near chance (0.5). A tail of heads reaches 0.76 (L10.H11), and one head reaches 0.90 on politics and sports relations, which are common topics in pretraining text.

## Post 2

Method: we feed in sequences of typed concepts and check whether the attention weight between two concepts predicts whether a real rule connects them.

We also use scrambled sequences, so word position cannot produce the signal. Only the identity of the types is left.

## Post 3

Heads that encode relations are known: Geva et al. describe "knowledge hub" heads that extract subject-attribute relations (arXiv:2304.14767).

We add a graph-level score against a full typed ontology, plus two tests: deleting the heads, and tracking them through training.

## Post 4  [attach: fig2_ablation_inert.png]

We fine-tuned the model to 100% rule-following, then zeroed out the top "graph heads."

After we deleted the best 8, rule-following was 99%. After we deleted 8 random heads, it was 99 to 100%.

The heads that best encode the rulebook contribute nothing measurable to following it.

Alt text: Bar chart titled "Deleting the 'graph heads' changes nothing." Rule-following with the enforcer off: no deletion 100%, top 8 graph heads deleted 99%, top 4 deleted 100%, two sets of 8 random heads deleted 99% and 100%.

## Post 5

Fine-tuning took the model from 13% to 100% rule-following and barely moved the attention map. Attention patterns changed by less than 0.02 AUC, pre-softmax q·k geometry by less than 0.01, and the top heads stayed the same.

The learning happened somewhere else in the network.

## Post 6  [attach: fig3_no_geometric_shadow.png]

A cleaner test: we trained on some regions of a 127-type merged ontology and held the others out.

Behavior went from 8% to 92% on trained regions and 4% to 16% on held-out ones. Attention's score for those same rules stayed flat (0.50 to 0.51 trained, 0.47 to 0.49 held out).

Alt text: Two panels titled "Behavior transformed. The attention map didn't move." Left: rule-valid output before and after training, 8% to 92% on trained regions and 4% to 16% on never-seen regions. Right: mean rule-detection AUC in attention for the same rules, 0.50 to 0.51 on trained regions and 0.47 to 0.49 on never-seen regions, both at chance.

## Post 7

Where did the learning go? Deletion only shows where it did not go. The likely place is the MLP and value pathways, where model-editing work locates facts (ROME, arXiv:2202.05262). That fits evidence that transformers solve structured tasks by pattern matching (arXiv:2305.18654).

## Post 8

The 0.90-AUC head knows politician and sports relations better than almost anything else in the network. In that same domain, the model generates valid sequences 4% of the time.

The model has this knowledge stored somewhere its generation does not use.

## Post 9

This may explain what many teams see in production: LLMs reason poorly over knowledge graphs natively and do better with SQL or schemas.

The internal graph is real but diffuse, and generation ignores it. Structure helps when it is enforced outside the weights (threads 1, 2).

## Post 10

Caveats: one model (Qwen2.5-1.5B, LoRA), one ontology family (DBpedia), attention-level probes only. Probing the value and MLP side comes next. These are deletion and training-dynamics results, not a full circuit analysis.

Code and data: github.com/MikeHLee/structure_of_clear_thinking

## Reply A (post as a reply to Post 7)

Background on where knowledge lives: MLPs as key-value memories (arXiv:2012.14913), facts editable in MLPs (ROME, arXiv:2202.05262), attention knowledge hubs (arXiv:2304.14767, @megamor2), relations as linear maps (arXiv:2308.09124). We add deletion and training dynamics.