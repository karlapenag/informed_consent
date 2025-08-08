import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter


def plot_distributions(profiles, output_path="plot_distributions.png"):
    df = pd.DataFrame(profiles)
    fields = ["literacy_level", "interest_level", "behavior"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, field in zip(axes.flatten(), fields):
        if field in df.columns:
            df[field].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Distribution of {field}")
            ax.set_xlabel(field)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)

def plot_top_feature_ratios_all_features(
    conversations,
    output_prefix="top5",
    min_total=3,
    top_k=5
):
    """
    aggregates ratios across literacy_level, behavior, and interest_level,
    then plots:
      - Top-K feature values most associated with consent
      - Top-K feature values most associated with no consent

    Assumes conversation["profile"] has the three fields and
    conversation["consent_decision"] is a string like:
      "the patient consented" OR "the patient did not consent"
    """

    # build df_feature_presence
    rows = []
    for conv in conversations:
        prof = conv.get("profile", {})
        if not prof:
            continue
        rows.append({
            "literacy_level": prof.get("literacy_level"),
            "behavior": prof.get("behavior"),
            "interest_level": prof.get("interest_level"),
            "consent_decision": (conv.get("consent_decision") or "").lower().strip()
        })

    if not rows:
        print("No conversations with profiles found; skipping plots.")
        return None

    df_combos = pd.DataFrame(rows)

    def _norm_decision(s: str) -> str:
        if "consented" in s:
            return "the patient consented"
        if "did not" in s and "consent" in s:
            return "the patient did not consent"
        return "unknown"

    df_combos["consent_decision"] = df_combos["consent_decision"].map(_norm_decision)

    # count occurrences by feature value & decision
    feature_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df_combos.iterrows():
        for feature in ["literacy_level", "behavior", "interest_level"]:
            key = f"{feature} = {row[feature]}"
            decision = row["consent_decision"]
            feature_counts[key][decision] += 1

    # flatten to a df with totals and ratios
    flat_data = []
    for feature_value, counts in feature_counts.items():
        total = sum(counts.values())
        if total < min_total:
            continue

        consented = counts.get("the patient consented", 0)
        not_consented = counts.get("the patient did not consent", 0)

        flat_data.append({
            "feature_value": feature_value,
            "total": total,
            "consented": consented,
            "did_not_consent": not_consented,
            "consented_ratio": consented / total if total else 0.0,
            "did_not_consent_ratio": not_consented / total if total else 0.0,
        })

    if not flat_data:
        print("No feature values reached the min_total threshold; skipping plots.")
        return None

    df_feature_presence = pd.DataFrame(flat_data)

    # Top-K: consent
    top_consent = df_feature_presence.sort_values(
        "consented_ratio", ascending=False
    ).head(top_k)

    plt.figure(figsize=(8, 5))
    plt.barh(top_consent["feature_value"], top_consent["consented_ratio"])
    plt.xlabel('Ratio of "patient consented"')
    plt.title("Top {} features most associated with consent".format(top_k))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_consented.png", dpi=300)
    plt.close()

    # Top-K: no consent
    top_no = df_feature_presence.sort_values(
        "did_not_consent_ratio", ascending=False
    ).head(top_k)

    plt.figure(figsize=(8, 5))
    plt.barh(top_no["feature_value"], top_no["did_not_consent_ratio"])
    plt.xlabel('Ratio of "patient did not consent"')
    plt.title("Top {} features most associated with no consent".format(top_k))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_no_consent.png", dpi=300)
    plt.close()

    # return table if need
    return df_feature_presence
