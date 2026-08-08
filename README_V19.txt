SR Studio v19 — Fixed 200 Training Workflow

AI screening workflow
1. Upload the full citation dataset (Title required; Abstract recommended).
2. Enter PICO/exclusion criteria.
3. SR Studio selects exactly 200 high-information training records:
   - 100 high PICO relevance
   - 70 decision-boundary records
   - 30 low PICO relevance
4. Download AI_Training_200.xlsx and label only Human_Label with O/X.
5. Upload the labeled 200-record file.
6. Train once and rank the full corpus. No iterative/extra labeling step is required.

Implementation notes
- Training-sample selection uses lightweight TF-IDF scoring for speed and does not require external models/APIs.
- Original row indices are retained internally so labels merge back to the full corpus accurately.
- Supervised screening uses cross-validation and a recall-oriented threshold.
- The UI reports current-project CV performance separately from the full-corpus ranking.
