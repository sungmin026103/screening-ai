# V36.2 — conservative small-error-bar patch

- In `figure_auto.py`, error-bar cap candidates are now checked for bilateral width, stem connectivity, and separation from the mean.
- The old fallback of interpreting an unverified topmost ink pixel as an error-bar endpoint is disabled for bars. If no verified cap is found, `error=None`, prompting manual confirmation.
- This prioritizes avoiding erroneous SD/SEM values over returning an automatic estimate for every bar.
- **Not yet validated on representative real-paper figures.** The patch is not evidence of improved extraction accuracy. Some valid caps may now be rejected.
- Existing manual digitizer and axis correction workflow remain available.
- Literature rationale: Pick et al. (2019), metaDigitise, DOI 10.1111/2041-210X.13118; Drevon et al. (2017), DOI 10.1177/0145445516673998.
