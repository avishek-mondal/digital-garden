---
layout: post
title: "Metrics in Field Extraction - what makes sense?"
date: 2026-02-01
categories: [Machine Learning, Engineering]
tags: [metrics, precision, recall, field-extraction]
---

How do people in key information extraction systems measure performance? How are metrics computed? This seems quite niche, but this is one of those areas in ML where the boring adds the most value. Most research papers and literature use standard classification metrics, but in my experience they don't always map cleanly to extraction tasks. Using them blindly can lead to misleading results.

<!-- more -->

Here are some notes on why the standard approach fails and a better way to calculate precision and recall for these specific problems.

## The standard definition trap

Usually, when we think of precision and recall, most researchers/practitioners (for example, [here](https://aihub.hkuspace.hku.hk/2025/09/03/document-intelligence-evolved-building-and-evaluating-kie-solutions-that-scale/)) rely on the following standard definitions:

* **True Positive (TP):** The field exists in the ground truth, and your system correctly extracted its values accordign to the field-specific comparator.
* **False Positive (FP):** Your system extracted a value for a field, but either the field doesn't exist in the ground truth, or the extracted value doesn't match the expected value.
* **False Negative (FN):** The field exists in the ground truth, but your system failed to extract it.
* **True Negative (TN):** The field doesn't exist in the ground truth, and your system correctly did not extract it.

### A motivating example
Let's look at a scenario extracting "Invoice Numbers" from four documents.

| Document | Ground Truth | Prediction | Result |
| :--- | :--- | :--- | :--- |
| doc1 | INV-100 | INV-100 | Match |
| doc2 | INV-200 | INV-999 | Mismatch |
| doc3 | INV-300 | INV-300 | Match |
| doc4 | INV-400 | INV-888 | Mismatch |

In this set, we have 2 matches and 2 mismatches. Under the standard definition, many engineers might calculate metrics like this:

1.  **TP = 2** (doc1, doc3)
2.  **FP = 2** (doc2, doc4) — *We predicted something, but it was wrong.*
3.  **FN = 0** — *We didn't "miss" the field; we just got the value wrong.*

The resulting metrics are:

* **Precision:** $2 / (2+2) = 50\%$
* **Recall:** $2 / (2+0) = 100\%$

### Why is this bad?

Is this really a "high-recall" model? A recall of **100%** gives the impression that the model will "usually contain the right invoice number somewhere". But in reality, half the time the number is completely wrong.

This approach optimizes for "detecting" the presence of a field rather than actually extracting the content correctly.

## A better approach: the double penalty

The key insight for field extraction is that **a mistake should increment both False Negatives and False Positives**.

If the model predicts `INV-999` when the truth is `INV-200`:

1.  You wrongly guessed a value (FP).
2.  You failed to detect the correct value (FN).

### New definitions
To formalize this, we can break down our states into more granular buckets:

* **gt_correct:** Ground truth exists, prediction matches.
* **gt_incorrect:** Ground truth exists, prediction mismatches.
* **gt_missing:** Ground truth exists, prediction is empty.
* **nogt_incorrect:** No ground truth, but model predicted something.
* **nogt_correct:** No ground truth, and model predicted nothing.

### Revised calculations
Using these states, we get formulas that actually reflect performance:

$$TP = \text{gt_correct}$$

$$FP = \text{gt_incorrect} + \text{nogt_incorrect}$$

$$FN = \text{gt_incorrect} + \text{gt_missing}$$

Notice that `gt_incorrect` now penalizes both precision and recall.

## One final note for practitioners

When building these systems, there is a tendency to distinguish between "The model detected the field but predicted 'None'" and "The model completely missed detecting the field."

**Does this distinction usually matter in practice? No.**

Treat these two cases as the same. A client or end-user effectively never cares about the distinction between a "detected None" and a "missing detection". Trying to keep them distinct is often a sign of premature optimization and will make the codebase significantly harder to maintain.
