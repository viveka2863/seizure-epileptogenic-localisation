# Project notes: how this went

A plain-language log of the process, for talking about the project. The README has the results; this has the reasoning.

## How this was built
I worked with Claude (Anthropic) as a tutor and coding assistant, and used Claude Code for the final cleanup, re-runs and write-up. I ran the notebooks and checked the numbers against each other. The decisions about what to trust and what to claim are written down below so I can explain them.

## The question
Kaggle's "Epileptic Seizure Recognition" has 500 EEG segments in five classes. I started with the hardest question in it: can a model tell Kaggle's class 2 ("tumour area") from class 3 ("healthy area")? I only have Kaggle's labels. The file cannot confirm what they mean, so I never claim anything about tumours.

## How the leak was found
- The CSV has 11,500 rows, but they are not independent: each segment is 23 one-second chunks, and chunks from the same segment are related (about 74% of the amplitude variance is explained by which segment a chunk belongs to). So I split by segment, not by row.
- Even with segment-level folds, a random forest scored 0.83 on raw values and 0.93 once overall level and spread were removed, for a problem that should be hard.
- An ablation pointed at the reason: shuffling the sample order inside each chunk dropped the forest to 0.52. It was using the exact waveform shape, which is also what you would use to match copies.
- That led me to compare segments with each other. Some are near-copies: the same recording under different IDs. A pair plotted on top of each other lines up almost perfectly (figure 06). About 44% of the class 2 and 3 segments had a partner above 0.8 correlation.
- Twins were always the same class. So a twin in training tells the model the answer for its partner in validation. That is leakage, even with segment-level splits.
- Fix: group twins and split by whole groups. With that, the same models fell from 0.57 to 0.93 down to 0.33 to 0.64. Chance on these folds spans 0.32 to 0.60, so the raw-value models were inside it and the best three (0.61 to 0.64) sat just above its top: a hint of signal, not proof.

## Decisions and why
- **Twin rule.** Link two segments if similarity is above 0.7, or if they sit in a tight cluster where every pair is above 0.4. After grouping, the most similar validation/training pair is about 0.64, well below the twin level. Later I tested stricter thresholds.
- **Shuffled-label range.** Instead of just saying "0.64 vs 0.5", I shuffled labels between whole groups 30 times to see what luck looks like. Chance is not a single number on a small dataset: it is 0.32 to 0.60 here.
- **Noise yardstick.** Re-making the folds moves scores by about 0.03. So differences under 0.03 are ties, and ties go to the simpler model. I fixed this rule before looking at the model comparisons.
- **Hold-out locked.** 100 segments set aside as whole groups. Their features were computed with everything else, but nothing was scored or inspected on them until the last step.
- **Seizure vs the rest.** Amplitude alone gives AUC 0.997. I stopped adding features because they added nothing. Accuracy hides the weak point: chance gets 0.80, and about 10 of 80 seizures are missed at the 0.5 cut-off. Those are quieter seizures. Imbalance fixes (class weights, oversampling, SMOTE) only move the cut-off; the ranking was already near perfect.
- **Segment features.** The first models looked at one chunk at a time and got close to nothing on class 2 vs 3. Rebuilding whole segments (all 23 chunks in order) and computing 23 features (8 about size, 15 about shape) gave a modest signal: about 0.70 to 0.79 depending on the features and the model.
- **Stress tests.** I tried to break the class 2 vs 3 result: stricter twin grouping (scores fell 0.02 to 0.04, stayed above chance), a negative control with power in the mains and noise bands (nothing), and a 40 Hz low-pass (little change). It survived all three. That does not make it true, only harder to dismiss.
- **Model comparison.** The SVM looked best (+0.044 over logistic on the saved folds). I did not believe it from one set of folds, so I made three fresh sets. The lead shrank to about +0.01 to +0.02. It was a lucky draw. Nested tuning also gave nothing. Logistic regression stays: simple and just as good.
- **Final checks.** Done once, models chosen in advance. Seizure vs classes 2 and 3 gave 0.988. Class 2 vs 3 gave 0.883, higher than development, on 40 segments only, and not a clean test because those segments had been used in earlier chunk-level work. I report it, but I do not lean on it.

## What I would do differently
- Look for duplicates in the very first audit, before any model. The leak cost me a whole first round of baselines (now in `notebooks/archive/`).
- Keep the splits in notebooks from the start, with a guard against overwriting them. A notebook that re-made the splits on every re-run was one slip away from changing them.
- Make the "how will I read this result" rule at the start. I did it late for some early notebooks.
- Plan the hold-out so that every question has a clean one. The class 2 vs 3 hold-out is not clean because it was drawn after earlier work.
- Find a dataset with patient IDs. The biggest open risk is that the models recognise a person or recording site, and this file cannot test that.

## Weak points I would say out loud
- Patient overlap cannot be ruled out, so the class 2 vs 3 AUC of about 0.75 may partly be identity.
- Twin grouping is a threshold. Weaker similarity may remain, so scores are upper bounds.
- 160 development segments and 40 in the hold-out: small. The final interval for class 2 vs 3 is 0.77 to 1.00.
- Gradient boosting was not tuned, and the SVM's built-in calibration uses ordinary (not twin-aware) folds inside the training data. It cannot leak into validation, but I noted it.
- The clips are short excerpts, not continuous recordings.

## Things not done
catch22, MiniROCKET, XGBoost/LightGBM, and any patient-level validation. They are listed as future work in the README.
