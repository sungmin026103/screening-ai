"""Offline smoke tests for V30 human-validation logic.
Run: python selftest_v30.py
"""
import numpy as np
import pandas as pd
import screening


def main():
    screening._HAS_SENTENCE_TRANSFORMERS = False
    n = 1000
    df = pd.DataFrame({
        "Title": [f"Rodent nutrition muscle study {i}" if i < 100 else f"Unrelated study {i}" for i in range(n)],
        "Abstract": [
            "mouse dietary intervention control grip strength muscle mass" if i < 100
            else "cell culture observational unrelated environmental outcome"
            for i in range(n)
        ],
    })
    criteria = "P: mouse rat rodent\nI: nutrition intervention\nC: control\nO: grip strength muscle mass"
    sample = screening.build_training_sample(df, criteria, "review\nin vitro", 200, 42)
    assert len(sample) == 200
    counts = sample["Training_Stratum"].value_counts().to_dict()
    # V33에서 상단 농축을 위해 배분을 70/20/10으로 바꿨다. 상수와 일치하는지로 검사한다.
    exp = [int(round(200 * a)) for a in screening.STRATUM_ALLOCATION]
    exp[-1] = 200 - sum(exp[:-1])
    assert counts.get("High PICO relevance") == exp[0]
    assert counts.get("Mid PICO relevance") == exp[1]
    assert counts.get("Low PICO relevance") == exp[2]
    assert sample["Validation_Record_ID"].nunique() == 200
    assert sample["Validation_Set_ID"].nunique() == 1
    assert abs(float(sample["Sampling_Weight"].sum()) - n) < 1e-3

    labelled = sample.copy()
    labelled["Human_Label"] = "X"
    labelled.loc[labelled.index[:20], "Human_Label"] = "O"
    canonical, stats = screening.validate_human_validation_file(sample, labelled)
    assert stats["complete"] and stats["labeled_n"] == 200
    assert stats["include_n"] == 20 and stats["exclude_n"] == 180

    bad = labelled.copy()
    bad.loc[0, "Human_Label"] = ""
    try:
        screening.validate_human_validation_file(sample, bad)
        raise AssertionError("Incomplete labels should be rejected")
    except ValueError:
        pass

    # Policy-evaluation smoke test: clearly separated scores should retain all positives.
    y = np.array(([1] * 40) + ([0] * 160))
    probs = np.r_[np.linspace(0.90, 0.99, 40), np.linspace(0.01, 0.25, 160)]
    folds = np.tile(np.arange(1, 6), 40)
    out = screening._crossfold_policy_evaluation(probs, y, folds, 0.95, weights=np.ones(200))
    assert out["safe_recall_weighted"] >= 0.95
    assert out["priority_recall_weighted"] >= 0.95

    print("V30 self-test: PASS")


if __name__ == "__main__":
    main()
