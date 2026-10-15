# MA feasibility predictor — LOSO replay validation report

Generated 2026-10-15 by `python3 -m feasibility.replay --full`. Aggregate numbers only; no titles or record-level data.

## Verdict (plain language)

**Overall: PASS (formal pass, fragile — see Robustness).** Gate metrics are computed on real outcomes (32 SR×outcome units from 6 SRs), using the nested leave-one-SR-out tuned results (each SR scored with tuning chosen on the other SRs).

| Criterion | Measured | Target | Verdict |
|---|---|---|---|
| 80% interval coverage | 87.4% (Wilson 95% CI 71.8%–95.0%) | 72–90% | **PASS** |
| 50% interval coverage | 54.1% (Wilson 95% CI 37.4%–70.0%) | 40–62% | **PASS** |
| Go/no-go accuracy (k ≥ 5, among non-보류) | 85.1% (Wilson 95% CI 69.1%–93.6%) | ≥ 85% | **PASS** |
| 보류 (abstain) share | 71.5% | reported | — |
| Median \|log((median+1)/(truth+1))\| | 0.511 | reported | — |

### Robustness of the verdict

- Decision accuracy clears the 85% bar by 0.1 points; its Wilson 95% CI is 69–94%.
- SR-equal-weighted accuracy is 72.2% (below 85%): the pooled figure is carried by SRs with many easy outcomes.
- The tool abstains (보류) on 71% of outcome predictions; the accuracy applies only to the remaining calls.
- Only 2 of 40 tuned designs pass under nested LOSO (nested accuracy range 65.4–85.1%), and passing is not monotone in n_total. The design itself is selected on the same 7 SRs (the tuning is nested, the design choice is not), so the chosen design's numbers carry a winner's-curse optimism.
- Per-SR 80% coverage outside 72–90%: L4 100%, L5 98%, L8 58%.

Chosen design: **n_total = 150**, census_top = 100, bounds = 0.01/0.05/0.2/1.0, single-phase, prior concentration cap = 5, q3 min-SR rule = off, decision thresholds go/no-go = 0.9/0.1 (source: nested-LOSO tuning).

## Headline metrics (chosen design, final run: 10,000 Monte Carlo draws)

| Set | units | cov50 pooled | cov50 SR-equal | cov80 pooled [Wilson] | cov80 SR-equal | cov95 | accuracy [Wilson] | accuracy SR-equal | 보류 | median width80 | median abs log ratio | labels | full texts read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real outcomes — deployment config, in-sample (optimistic) | 32 | 51.7% | 49.9% | 86.2% [70.4%–94.3%] | 84.5% | 95.9% | 85.5% [69.6%–93.8%] | 73.4% | 71.3% | 30.0 | 0.636 | 150 | 7.0 |
| Real outcomes, nested-LOSO tuned (2,000 draws) — gate | 32 | 54.1% | 52.8% | 87.4% [71.8%–95.0%] | 85.7% | 96.7% | 85.1% [69.1%–93.6%] | 72.2% | 71.5% | 30.0 | 0.511 | 150 | 7.0 |
| Pseudo-outcome _included (stages 1–2) | 7 | 56.9% | 56.9% | 81.2% [44.4%–95.9%] | 81.2% | 94.9% | 90.7% [53.8%–98.8%] | 88.8% | 51.6% | 48.0 | 0.551 | 150 | 6.2 |

Wilson intervals use the number of SR×outcome units as n (replications of one unit are not independent). They are still optimistic: outcomes within an SR are correlated, and the effective sample is closer to the number of SRs.

## Method

- **Library.** 8 SRs; 7 with complete ground truth (L1, L2, L3, L4, L5, L6, L8); excluded (ground_truth_complete = false): L7 (full-text decisions known: False, undecided full texts: 11). Without outcome k (only the `_included` pseudo-outcome): L3.
- **Leave-one-SR-out.** For each target SR, priors (stratum z1 prevalence, q2, q3 families) are fitted on the other SRs only (`fit_priors(exclude=target)`), once per (SR, bounds, cap). Relevance scores are computed once per SR from its own PICO text (title + abstract where enriched).
- **Replication.** 200 replications per SR per design with seeds SeedSequence([SR seed, rep]) shared across designs (paired comparison). Each replication: stratify by relevance rank, allocate (S1 census up to census_top, rest by weights), draw the sample; two-phase designs label 70% first and allocate the remaining 30% by `neyman_followup` among unsampled records. Revealed: z1 for sampled records; for sampled passers, full-text result (included / excluded / unobtainable; z2 undecided counts as unobtainable).
- **Truth.** Real outcome o: the SR's final k_o (clipped to the number of included records where larger). Which included records carry data for o is not recorded in the library, so each replication assigns k_o of the true included records at random (without replacement) and counts those among the sampled read includes (quant_o). Pseudo-outcome `_included`: truth = number of records with z2 = 1 (unobtainable full texts are not part of the truth), estimated with quant = y and q3 prior Beta(1e6, 1).
- **Clipping.** No outcome k exceeded the included count.
- **Metrics.** Coverage at 50/80/95% (inclusive integer intervals), pooled over SR×outcome×replication and SR-equal-weighted (mean of per-SR coverage); go/no-go accuracy at threshold k = 5 among non-보류 calls (진행 correct iff truth ≥ 5, 비진행 correct iff truth < 5); 보류 share; median 80% width; median |log((median+1)/(truth+1))|; mean labels and full texts read.
- **Design grid.** n_total ∈ {100,150,200,250,300} × census_top ∈ {50,100} × bounds ∈ {A = 0.01/0.05/0.20/1.0, B = 0.02/0.10/0.30/1.0} × {single-phase, two-phase 70/30}; 40 designs, 2,000 draws each, cap 20, go/no-go 0.8/0.2. Selection rule (fixed before looking): smallest n_total meeting all three criteria, then smaller census_top, single-phase, bounds A. Two-phase uses one split (70/30): enough first-phase labels (≥ 70) for stable stratum rates before Neyman reallocation, while leaving a follow-up budget large enough to matter.

## Grid summary (real outcomes, cap 20, go/no-go 0.8/0.2, 2,000 draws)

| design | cov50 | cov80 | cov80_sr | acc | acc_sr | abstain | width80_med | alog_med | inc_cov50 | inc_cov80 | inc_acc | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n100_c50_A_1ph | 0.535 | 0.785 | 0.795 | 0.672 | 0.679 | 0.376 | 58.000 | 0.619 | 0.464 | 0.829 | 0.922 | False |
| n100_c50_A_2ph | 0.478 | 0.746 | 0.765 | 0.663 | 0.654 | 0.409 | 52.000 | 0.657 | 0.456 | 0.849 | 0.968 | False |
| n100_c50_B_1ph | 0.577 | 0.819 | 0.829 | 0.707 | 0.688 | 0.325 | 49.000 | 0.523 | 0.543 | 0.876 | 0.946 | False |
| n100_c50_B_2ph | 0.525 | 0.780 | 0.798 | 0.715 | 0.668 | 0.324 | 46.000 | 0.523 | 0.537 | 0.889 | 0.973 | False |
| n100_c100_A_1ph | 0.498 | 0.791 | 0.824 | 0.616 | 0.689 | 0.262 | 102.000 | 0.693 | 0.339 | 0.802 | 0.920 | False |
| n100_c100_A_2ph | 0.463 | 0.738 | 0.763 | 0.593 | 0.661 | 0.279 | 66.000 | 0.619 | 0.427 | 0.872 | 0.968 | False |
| n100_c100_B_1ph | 0.549 | 0.790 | 0.810 | 0.641 | 0.704 | 0.229 | 79.000 | 0.629 | 0.476 | 0.877 | 0.946 | False |
| n100_c100_B_2ph | 0.504 | 0.770 | 0.788 | 0.633 | 0.664 | 0.247 | 56.000 | 0.511 | 0.564 | 0.864 | 0.973 | False |
| n150_c50_A_1ph | 0.553 | 0.821 | 0.821 | 0.697 | 0.676 | 0.327 | 42.000 | 0.588 | 0.572 | 0.856 | 0.947 | False |
| n150_c50_A_2ph | 0.525 | 0.787 | 0.794 | 0.680 | 0.668 | 0.355 | 36.000 | 0.588 | 0.579 | 0.833 | 0.981 | False |
| n150_c50_B_1ph | 0.580 | 0.820 | 0.833 | 0.736 | 0.701 | 0.276 | 38.000 | 0.486 | 0.587 | 0.861 | 0.953 | False |
| n150_c50_B_2ph | 0.569 | 0.811 | 0.821 | 0.710 | 0.691 | 0.301 | 34.000 | 0.525 | 0.599 | 0.873 | 0.978 | False |
| n150_c100_A_1ph | 0.580 | 0.801 | 0.810 | 0.700 | 0.682 | 0.359 | 57.000 | 0.511 | 0.530 | 0.861 | 0.950 | False |
| n150_c100_A_2ph | 0.473 | 0.754 | 0.778 | 0.716 | 0.678 | 0.341 | 53.000 | 0.531 | 0.502 | 0.817 | 0.982 | False |
| n150_c100_B_1ph | 0.592 | 0.825 | 0.838 | 0.716 | 0.703 | 0.283 | 46.000 | 0.454 | 0.568 | 0.864 | 0.955 | False |
| n150_c100_B_2ph | 0.530 | 0.779 | 0.804 | 0.736 | 0.698 | 0.275 | 45.000 | 0.486 | 0.581 | 0.884 | 0.978 | False |
| n200_c50_A_1ph | 0.528 | 0.831 | 0.834 | 0.716 | 0.692 | 0.296 | 33.000 | 0.486 | 0.584 | 0.870 | 0.981 | False |
| n200_c50_A_2ph | 0.518 | 0.824 | 0.829 | 0.709 | 0.678 | 0.318 | 28.000 | 0.516 | 0.532 | 0.857 | 0.993 | False |
| n200_c50_B_1ph | 0.553 | 0.809 | 0.822 | 0.743 | 0.699 | 0.273 | 31.000 | 0.405 | 0.561 | 0.847 | 0.963 | False |
| n200_c50_B_2ph | 0.553 | 0.820 | 0.834 | 0.729 | 0.692 | 0.270 | 27.000 | 0.425 | 0.575 | 0.864 | 0.992 | False |
| n200_c100_A_1ph | 0.572 | 0.833 | 0.839 | 0.706 | 0.688 | 0.299 | 41.000 | 0.511 | 0.605 | 0.888 | 0.981 | False |
| n200_c100_A_2ph | 0.540 | 0.805 | 0.806 | 0.712 | 0.683 | 0.319 | 35.000 | 0.588 | 0.558 | 0.825 | 0.993 | False |
| n200_c100_B_1ph | 0.571 | 0.836 | 0.853 | 0.732 | 0.704 | 0.250 | 35.000 | 0.442 | 0.595 | 0.869 | 0.962 | False |
| n200_c100_B_2ph | 0.583 | 0.825 | 0.843 | 0.720 | 0.698 | 0.273 | 31.000 | 0.470 | 0.600 | 0.889 | 0.993 | False |
| n250_c50_A_1ph | 0.521 | 0.832 | 0.836 | 0.738 | 0.695 | 0.295 | 28.000 | 0.405 | 0.586 | 0.878 | 0.992 | False |
| n250_c50_A_2ph | 0.510 | 0.820 | 0.834 | 0.727 | 0.684 | 0.292 | 23.000 | 0.438 | 0.554 | 0.854 | 0.994 | False |
| n250_c50_B_1ph | 0.557 | 0.813 | 0.825 | 0.756 | 0.703 | 0.275 | 27.000 | 0.353 | 0.566 | 0.847 | 0.987 | False |
| n250_c50_B_2ph | 0.540 | 0.815 | 0.833 | 0.749 | 0.698 | 0.273 | 24.000 | 0.368 | 0.538 | 0.840 | 0.993 | False |
| n250_c100_A_1ph | 0.549 | 0.848 | 0.850 | 0.737 | 0.697 | 0.278 | 33.000 | 0.466 | 0.569 | 0.873 | 0.992 | False |
| n250_c100_A_2ph | 0.541 | 0.813 | 0.828 | 0.721 | 0.684 | 0.301 | 27.000 | 0.470 | 0.574 | 0.844 | 0.994 | False |
| n250_c100_B_1ph | 0.557 | 0.827 | 0.842 | 0.753 | 0.706 | 0.264 | 29.000 | 0.357 | 0.588 | 0.865 | 0.987 | False |
| n250_c100_B_2ph | 0.549 | 0.830 | 0.846 | 0.749 | 0.701 | 0.272 | 25.000 | 0.368 | 0.586 | 0.867 | 0.993 | False |
| n300_c50_A_1ph | 0.527 | 0.841 | 0.843 | 0.760 | 0.711 | 0.307 | 25.000 | 0.446 | 0.547 | 0.871 | 0.985 | False |
| n300_c50_A_2ph | 0.513 | 0.810 | 0.822 | 0.749 | 0.694 | 0.296 | 21.000 | 0.375 | 0.559 | 0.853 | 0.994 | False |
| n300_c50_B_1ph | 0.539 | 0.820 | 0.837 | 0.758 | 0.705 | 0.252 | 24.000 | 0.336 | 0.579 | 0.843 | 0.992 | False |
| n300_c50_B_2ph | 0.542 | 0.815 | 0.835 | 0.755 | 0.700 | 0.259 | 21.000 | 0.336 | 0.539 | 0.844 | 0.997 | False |
| n300_c100_A_1ph | 0.533 | 0.838 | 0.839 | 0.744 | 0.707 | 0.285 | 28.000 | 0.385 | 0.561 | 0.872 | 0.985 | False |
| n300_c100_A_2ph | 0.512 | 0.817 | 0.828 | 0.740 | 0.696 | 0.302 | 23.000 | 0.405 | 0.553 | 0.850 | 0.994 | False |
| n300_c100_B_1ph | 0.562 | 0.828 | 0.846 | 0.763 | 0.713 | 0.261 | 25.000 | 0.336 | 0.599 | 0.854 | 0.992 | False |
| n300_c100_B_2ph | 0.548 | 0.819 | 0.830 | 0.766 | 0.711 | 0.288 | 22.000 | 0.336 | 0.573 | 0.845 | 0.997 | False |

## Tuning (no grid design passed)

Allowed knobs only: prior concentration cap ∈ {5,10,20,40}, q3 min-SR rule ∈ {off, 3} (a q3 family fitted from < 3 SRs falls back to _default), decision thresholds ∈ {0.8/0.2, 0.85/0.15, 0.9/0.1} — 24 configurations, applied to all 40 grid designs (single- and two-phase). **Nested LOSO:** for each held-out SR the configuration is chosen on the other SRs (passing configurations first, then smallest distance to the acceptance region) and only the held-out SR's results under that choice are pooled. Caveat: in the inner loop each other SR's predictions used priors that still contained the held-out SR (single-level LOSO priors inside), a mild optimistic bias. The deployment configuration (`full_cfg`) applies the same rule to all SRs; its own in-sample metrics are optimistic and are not used for the verdict.

| design | nested_cov50 | nested_cov80 | nested_acc | nested_abstain | pass | full_cfg |
|---|---|---|---|---|---|---|
| n100_c50_A_1ph | 0.517 | 0.869 | 0.826 | 0.733 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n100_c50_A_2ph | 0.537 | 0.848 | 0.748 | 0.736 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n100_c50_B_1ph | 0.545 | 0.867 | 0.805 | 0.662 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n100_c50_B_2ph | 0.558 | 0.865 | 0.787 | 0.676 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n100_c100_A_1ph | 0.620 | 0.758 | 0.654 | 0.572 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n100_c100_A_2ph | 0.623 | 0.817 | 0.713 | 0.661 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n100_c100_B_1ph | 0.646 | 0.802 | 0.685 | 0.567 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n100_c100_B_2ph | 0.625 | 0.777 | 0.680 | 0.533 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n150_c50_A_1ph | 0.542 | 0.830 | 0.845 | 0.653 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n150_c50_A_2ph | 0.534 | 0.831 | 0.816 | 0.669 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n150_c50_B_1ph | 0.553 | 0.835 | 0.798 | 0.568 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n150_c50_B_2ph | 0.547 | 0.831 | 0.795 | 0.608 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n150_c100_A_1ph | 0.541 | 0.874 | 0.851 | 0.715 | True | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n150_c100_A_2ph | 0.604 | 0.876 | 0.845 | 0.708 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n150_c100_B_1ph | 0.550 | 0.886 | 0.836 | 0.612 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n150_c100_B_2ph | 0.592 | 0.896 | 0.826 | 0.623 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c50_A_1ph | 0.526 | 0.787 | 0.848 | 0.609 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c50_A_2ph | 0.558 | 0.784 | 0.810 | 0.606 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c50_B_1ph | 0.521 | 0.808 | 0.793 | 0.492 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c50_B_2ph | 0.556 | 0.800 | 0.787 | 0.522 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c100_A_1ph | 0.546 | 0.849 | 0.851 | 0.622 | True | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c100_A_2ph | 0.536 | 0.838 | 0.824 | 0.631 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c100_B_1ph | 0.537 | 0.822 | 0.815 | 0.530 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n200_c100_B_2ph | 0.564 | 0.816 | 0.789 | 0.544 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c50_A_1ph | 0.542 | 0.783 | 0.815 | 0.526 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c50_A_2ph | 0.539 | 0.795 | 0.802 | 0.529 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c50_B_1ph | 0.520 | 0.801 | 0.802 | 0.482 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c50_B_2ph | 0.535 | 0.805 | 0.789 | 0.505 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n250_c100_A_1ph | 0.559 | 0.812 | 0.844 | 0.573 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n250_c100_A_2ph | 0.570 | 0.809 | 0.802 | 0.548 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c100_B_1ph | 0.522 | 0.808 | 0.809 | 0.484 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n250_c100_B_2ph | 0.548 | 0.802 | 0.795 | 0.515 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n300_c50_A_1ph | 0.517 | 0.786 | 0.821 | 0.502 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n300_c50_A_2ph | 0.540 | 0.792 | 0.803 | 0.503 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n300_c50_B_1ph | 0.499 | 0.800 | 0.817 | 0.469 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n300_c50_B_2ph | 0.522 | 0.806 | 0.804 | 0.474 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n300_c100_A_1ph | 0.520 | 0.799 | 0.818 | 0.505 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n300_c100_A_2ph | 0.530 | 0.775 | 0.797 | 0.509 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |
| n300_c100_B_1ph | 0.501 | 0.802 | 0.820 | 0.472 | False | cap=5.0 min_sr=0 go/nogo=0.9/0.1 |
| n300_c100_B_2ph | 0.507 | 0.806 | 0.808 | 0.486 | False | cap=5.0 min_sr=3 go/nogo=0.9/0.1 |

Nested choice per held-out SR (chosen design):

| held_out | cap | min_sr | go | nogo | inner_pass |
|---|---|---|---|---|---|
| L1 | 5.000 | 3 | 0.900 | 0.100 | True |
| L2 | 5.000 | 0 | 0.900 | 0.100 | True |
| L3 | 5.000 | 0 | 0.900 | 0.100 | True |
| L4 | 5.000 | 3 | 0.900 | 0.100 | True |
| L5 | 5.000 | 0 | 0.900 | 0.100 | True |
| L6 | 5.000 | 3 | 0.900 | 0.100 | False |
| L8 | 5.000 | 0 | 0.900 | 0.100 | False |

## Per-SR breakdown (chosen design, final run)

Real outcomes:

| sr_id | n_outcomes | truth_range | cov50 | cov80 | cov95 | acc | abstain | width80_med | alog_med | labels_mean | ft_read_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 4 | 3-24 | 0.372 | 0.755 | 0.946 | 0.951 | 0.899 | 14.000 | 1.179 | 150.000 | 4.350 |
| L2 | 8 | 4-53 | 0.450 | 0.897 | 0.964 | 0.634 | 0.741 | 90.000 | 1.030 | 150.000 | 2.415 |
| L4 | 4 | 15-15 | 0.374 | 0.998 | 1.000 | 1.000 | 0.924 | 34.000 | 0.981 | 150.000 | 1.405 |
| L5 | 4 | 4-4 | 0.845 | 0.978 | 0.994 | 0.000 | 0.978 | 31.000 | 0.223 | 150.000 | 0.415 |
| L6 | 9 | 1-12 | 0.620 | 0.861 | 0.964 | 0.936 | 0.425 | 6.000 | 0.262 | 150.000 | 24.580 |
| L8 | 3 | 13-16 | 0.332 | 0.585 | 0.847 | 0.885 | 0.622 | 13.000 | 0.693 | 150.000 | 8.855 |

Pseudo-outcome `_included`:

| sr_id | n_outcomes | truth_range | cov50 | cov80 | cov95 | acc | abstain | width80_med | alog_med | labels_mean | ft_read_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | 39-39 | 0.290 | 0.635 | 1.000 | 1.000 | 0.505 | 41.000 | 0.981 | 150.000 | 4.350 |
| L2 | 1 | 58-58 | 0.495 | 0.975 | 1.000 | 1.000 | 0.505 | 145.500 | 1.033 | 150.000 | 2.415 |
| L3 | 1 | 6-6 | 0.525 | 0.640 | 0.710 | 0.212 | 0.600 | 17.000 | 0.589 | 150.000 | 1.700 |
| L4 | 1 | 17-17 | 0.890 | 0.960 | 1.000 | 1.000 | 0.775 | 64.000 | 0.847 | 150.000 | 1.405 |
| L5 | 1 | 7-7 | 0.870 | 0.975 | 0.995 | 1.000 | 0.940 | 50.000 | 0.288 | 150.000 | 0.415 |
| L6 | 1 | 19-19 | 0.520 | 0.815 | 0.970 | 1.000 | 0.000 | 10.000 | 0.182 | 150.000 | 24.580 |
| L8 | 1 | 20-20 | 0.390 | 0.685 | 0.970 | 1.000 | 0.290 | 23.000 | 0.549 | 150.000 | 8.855 |

Per outcome (real outcomes; accuracy among called; p_go / p_nogo = share of replications):

| sr_id | outcome | truth | median_of_median | cov50 | cov80 | p_go | p_nogo | p_abstain | acc_called | width80_med |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | muscle | 12 | 3.000 | 0.310 | 0.615 | 0.090 | 0.000 | 0.910 | 1.000 | 17.000 |
| L1 | bone | 24 | 6.000 | 0.255 | 0.470 | 0.255 | 0.000 | 0.745 | 1.000 | 21.000 |
| L1 | 미토콘드리아 | 3 | 1.000 | 0.575 | 0.960 | 0.020 | 0.000 | 0.980 | 0.000 | 11.000 |
| L1 | 산화스트레스 | 5 | 1.000 | 0.350 | 0.975 | 0.040 | 0.000 | 0.960 | 1.000 | 11.000 |
| L2 | TG | 53 | 14.000 | 0.285 | 0.990 | 0.345 | 0.000 | 0.655 | 1.000 | 103.000 |
| L2 | TC | 51 | 14.000 | 0.280 | 0.985 | 0.345 | 0.000 | 0.655 | 1.000 | 103.500 |
| L2 | LDL | 44 | 14.000 | 0.355 | 0.975 | 0.305 | 0.000 | 0.695 | 1.000 | 100.000 |
| L2 | HDL | 46 | 14.000 | 0.330 | 0.985 | 0.320 | 0.000 | 0.680 | 1.000 | 101.000 |
| L2 | TG (clinical) | 4 | 11.000 | 0.565 | 0.815 | 0.185 | 0.000 | 0.815 | 0.000 | 81.000 |
| L2 | TC (clinical) | 4 | 10.000 | 0.595 | 0.800 | 0.200 | 0.000 | 0.800 | 0.000 | 80.000 |
| L2 | LDL (clinical) | 4 | 10.000 | 0.610 | 0.805 | 0.195 | 0.000 | 0.805 | 0.000 | 79.000 |
| L2 | HDL (clinical) | 4 | 10.500 | 0.580 | 0.820 | 0.180 | 0.000 | 0.820 | 0.000 | 79.500 |
| L4 | TG | 15 | 5.000 | 0.350 | 0.995 | 0.075 | 0.000 | 0.925 | 1.000 | 34.000 |
| L4 | TC | 15 | 5.000 | 0.370 | 0.995 | 0.080 | 0.000 | 0.920 | 1.000 | 34.000 |
| L4 | LDL | 15 | 5.000 | 0.385 | 1.000 | 0.075 | 0.000 | 0.925 | 1.000 | 34.000 |
| L4 | HDL | 15 | 5.000 | 0.390 | 1.000 | 0.075 | 0.000 | 0.925 | 1.000 | 34.000 |
| L5 | TG | 4 | 3.000 | 0.850 | 0.980 | 0.020 | 0.000 | 0.980 | 0.000 | 31.000 |
| L5 | TC | 4 | 3.000 | 0.825 | 0.980 | 0.020 | 0.000 | 0.980 | 0.000 | 31.000 |
| L5 | LDL-c | 4 | 3.000 | 0.855 | 0.975 | 0.025 | 0.000 | 0.975 | 0.000 | 31.000 |
| L5 | HDL-c | 4 | 3.000 | 0.850 | 0.975 | 0.025 | 0.000 | 0.975 | 0.000 | 31.000 |
| L6 | WAT | 12 | 11.000 | 0.580 | 0.830 | 0.945 | 0.000 | 0.055 | 1.000 | 8.000 |
| L6 | TG | 12 | 11.000 | 0.525 | 0.840 | 0.930 | 0.000 | 0.070 | 1.000 | 8.000 |
| L6 | TC | 12 | 11.000 | 0.525 | 0.820 | 0.955 | 0.000 | 0.045 | 1.000 | 8.000 |
| L6 | UCP1 | 4 | 3.000 | 0.610 | 0.795 | 0.020 | 0.240 | 0.740 | 0.923 | 4.000 |
| L6 | FAS | 5 | 5.000 | 0.530 | 0.850 | 0.170 | 0.090 | 0.740 | 0.654 | 5.000 |
| L6 | PPARα | 5 | 4.000 | 0.555 | 0.830 | 0.135 | 0.130 | 0.735 | 0.509 | 5.000 |
| L6 | Hepatic TG | 7 | 7.000 | 0.685 | 0.885 | 0.550 | 0.010 | 0.440 | 0.982 | 6.000 |
| L6 | Adipocyte size | 4 | 4.000 | 0.670 | 0.895 | 0.080 | 0.125 | 0.795 | 0.610 | 5.000 |
| L6 | muscle mass | 1 | 1.000 | 0.900 | 1.000 | 0.000 | 0.795 | 0.205 | 1.000 | 3.000 |
| L8 | 동역학 | 16 | 7.000 | 0.295 | 0.485 | 0.325 | 0.035 | 0.640 | 0.903 | 11.500 |
| L8 | grip | 13 | 8.500 | 0.375 | 0.690 | 0.410 | 0.015 | 0.575 | 0.965 | 15.000 |
| L8 | CSA | 13 | 7.000 | 0.325 | 0.580 | 0.270 | 0.080 | 0.650 | 0.771 | 13.000 |

## Where decisions fail

Wrong calls in the final run: 266 of 1837 non-보류 calls. Largest contributors (SR, outcome, true k, wrong calls, share of all wrong calls):

| sr_id | outcome | truth | wrong | called | share_of_errors |
|---|---|---|---|---|---|
| L2 | TC (clinical) | 4 | 40.000 | 40 | 0.150 |
| L2 | LDL (clinical) | 4 | 39.000 | 39 | 0.147 |
| L2 | TG (clinical) | 4 | 37.000 | 37 | 0.139 |
| L2 | HDL (clinical) | 4 | 36.000 | 36 | 0.135 |
| L6 | PPARα | 5 | 26.000 | 53 | 0.098 |
| L6 | FAS | 5 | 18.000 | 52 | 0.068 |
| L8 | CSA | 13 | 16.000 | 70 | 0.060 |
| L6 | Adipocyte size | 4 | 16.000 | 41 | 0.060 |

## Limitations

- **Few validation units.** 8 SRs in the library, 7 usable (L7 excluded); L3 without outcome k, so real-outcome metrics rest on 6 SRs. Outcomes within an SR are strongly correlated (e.g. TG/TC/LDL/HDL share the same included set), so the Wilson intervals computed with SR×outcome units as n understate the uncertainty.
- **Titles only for L2, L3, L4, L5.** Their abstracts are mostly missing (abstract match rate 0.3–5.4%), so relevance scoring and stratification use titles alone; real use has abstracts, so stratification in practice should be at least as sharp.
- **Replications.** 200 replications per SR per design instead of the 1,000 seeds in spec §7, to keep the 40-design grid plus nested tuning (40 designs × 8 prior variants) within the runtime budget. Monte Carlo error of a pooled coverage at 200 reps × ≥ 3 units is ≈ 1–2 points, well below the between-SR spread, which dominates the uncertainty (see Wilson CIs).
- **User labels = ground truth.** The replay reveals the SR's recorded decisions; reviewer disagreement and label noise of a new screening are not modelled.
- **Exchangeability in outcome assignment.** Which included records report each outcome is unknown, so k_o of them are drawn at random each replication, independently across outcomes. Real outcome data cluster (the same papers report all lipid endpoints), so real q3 sampling variability may differ.
- **Library domain.** Nutrition / preclinical animal SRs. A future SR in a different field (e.g. clinical RCTs) may not match the priors; the clinical sub-outcomes of L2 illustrate this failure mode.
- **Single fixed threshold.** Accuracy is evaluated at k = 5 only.

Runtime: this invocation 35.1 min with 2 worker(s); recorded stage times (cached stages are reused from --grid-cache/--tune-cache): grid 6.9 min, tuning 59.4 min, final 0.5 min.
