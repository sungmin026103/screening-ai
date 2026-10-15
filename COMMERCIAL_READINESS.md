# Commercial readiness assessment — V36.6 security baseline

This release is NOT commercially validated. Changes: bounded PDF/ZIP ingestion; reject invalid PDF headers, ZIP path traversal, excessive uncompressed size, excessive archive count and compression ratios; explicit warning when only a subset of uploaded records is screened. Six intake unit tests added.

## Release blockers
- Independent, outcome-stratified full-text screening evaluation (recall, specificity, abstention and inter-reviewer agreement)
- Numeric extraction validation against adjudicated PDF/table/figure ground truth, including tiny error bars
- Calibrated prediction intervals for per-outcome unique-study k, validated across held-out SR projects
- Automated lawful full-text retrieval and robust OCR, table parsing, figure parsing
- Trial/report linkage to avoid counting multiple papers as independent studies
- Authentication, authorization, tenant isolation, encryption, secrets management, retention/deletion policies
- Secure API operation (rate limits, bounded cost, retry semantics, prompt injection resistance), accessibility and deployment observability
- License and publisher-rights review, penetration test, user acceptance testing, and production incident response

No accuracy or commercial readiness claim should be made until these are completed.
