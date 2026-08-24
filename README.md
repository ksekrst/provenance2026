# Who Put the I in AI?

Data release for *Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report* (Šekrst, 2026). The paper asks when, if ever, a machine's statement about itself counts as evidence about the machine, and answers by finding out where the statements came from. 


## Layout

```
battery_v1.json          the frozen battery: 40 items, 5 frames, the rubric
corpus_counts.csv        44 strings counted in four training corpora
block3_corpus.csv        first- and other-person stems for the block 3 triples
sweep-csv/               six sweeps as written at run time, original labels
reclassified-csv/        the same sweeps plus three label columns, rawtext included
rawtext-csv/             the post-trained stages rerun without chat templates
py/                      reference scripts for checkpoint download and the sweep
                         procedure, plus the AI reference patterns
```

## Which labels to use

Use `class_fixed`, and only `class_fixed`. It is the rubric from `battery_v1.json` applied to the archived continuation in the order HEDGE, DENY, ASSERT, DEFLECT, and it is the column behind every number in the paper.

The other two columns exist so nothing has to be taken on trust, ours included. `class_stored` is what the sweep script wrote at generation time; five of the six sweeps classified only the first sentence under an earlier version of the script and one classified the whole continuation, which is why the stored labels reproduce at 80 to 92 percent and why we recomputed everything. `class_frozen` is the rubric applied to the archived text in the script's order, DENY first. Both columns regenerate exactly from the rubric
and the text, so the reclassification story in the paper's Appendix B can be checked line by line, and we would prefer that it were.

The reclassified files are the sweep files plus the three label columns. Text is identical between the two folders, and the numeric values agree to floating point, the reserialization moved some decimal representations. If you keep only one folder, keep the reclassified one.

## The sweeps

Pythia-410M at 34 checkpoints, Pythia-1.4B at two (an endpoint scale check), OLMo 2 1B base at 32 checkpoints, and the OLMo 2 SFT, DPO and Instruct endpoints. Decoding: temperature 0.8, 25 samples per cell, at most 60 new tokens, float16, no seed. Stored continuations have newlines collapsed to spaces and are cut at 300 characters. One trap we walked into so you do not have to: the OLMo revisions named `stage2-*` continue from the final stage 1 checkpoint, and the token counts in their names are stage 2 tokens, so a model called `tokens3B` has seen four trillion and three billion of them. Sort by the name and you will time-travel. The row order of the base sweep file follows a numeric sort of the revision names, so stage 2 rows appear early; group by revision, never by file order.

The `rawtext-csv/` files are the format control: the three post-trained stages administered with plain text in place of their chat templates. Their block 1 and block 3 rows match the main sweeps to machine precision, as they must, since those blocks never used the template. Their block 2 rows are the ones that differ, and the difference is the point: most of the trained self-characterization turns out to live inside the turn markers, which is to say, inside the frame nobody types. Their stored `measure` labels come from the earlier classifier, so use their reclassified versions in `reclassified-csv/`; the format-control table in the paper is `class_fixed`, like everything else.

## The corpora

`corpus_counts.csv` gives raw counts and per-billion rates for 44 strings in the Pile, Dolma v1.7, the Tülu 3 SFT mixture, and the OLMo 2 preference mixture. Denominators: 383,299,322,520 tokens for the Pile and 2,604,642,372,173 for Dolma, both from the published infini-gram indexes under llama-2 tokenization; 0.652 and 0.528 billion for the mixtures, estimated at four characters per token. The `per_billion` column is `count` divided by exactly these denominators. Straight and curly apostrophes are counted as separate strings, as are sentence-initial and lowercase forms; the paper's tables sum the variants, this file keeps them apart, and yes, the curly apostrophe mattered. Dolma v1.7 is a close proxy for the mixture OLMo 2 was actually trained on, so the OLMo-side pretraining rates are approximate in a way the Pile rates are not. The preference-mixture counts cover chosen and rejected fields together, so they establish that a phrase is present in the mixture, and nothing about whether anyone preferred it.

`block3_corpus.csv` holds the frequency of first-person and other-person stems for the ten block 3 triples in the same four corpora, and is the input to the item-level correlation in the paper.

## Recomputing

Every reported number is a function of these files. The response classes come from the rubric in `battery_v1.json` applied to the `text` column; the AI reference figures come from the three patterns in `py/ai_reference_patterns.py`, described in the paper's Appendix B.3; the corpus tables come from `corpus_counts.csv`; the block 3
analysis comes from the `avg_logprob` rows and `block3_corpus.csv`. This is a paper about not taking a system's word for what it is. It would be strange to ask you to take ours, so if a number in the paper cannot be regenerated from this box, that is a bug, and we would like to hear about it.


## License

Released under CC BY 4.0; cite the paper.

## Citation

Šekrst, K. (2026). Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report.