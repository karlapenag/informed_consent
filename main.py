import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from openai import OpenAI
from profiles import generate_patient_profile
from pdf_reader import extract_text_without_header_footer_and_replace
from gpt_conversations import create_pretraining_dataset
from evaluator_function_edited import save_my_analysis, METRICS
from plotting import (
    plot_distributions,
    plot_categorical_distributions,
    plot_top_feature_ratios_all_features
)

def _to_py(obj):
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(x) for x in obj]
    if isinstance(obj, np.generic):     
        return obj.item()
    if isinstance(obj, np.ndarray):     
        return obj.tolist()
    return obj

def load_or_generate_profiles(path, build_count):
    if path:
        with open(path) as f:
            return json.load(f)
    else:
        profiles = [generate_patient_profile() for _ in range(build_count)]
        filename = "patient_profiles.json"
        if os.path.exists(filename):
            suffix = datetime.now().strftime("_%m%d%Y") 
            filename = filename.replace(".json", f"{suffix}.json")
        with open(filename, "w") as f:
            json.dump(_to_py(profiles), f, indent=2)     
        return profiles


def main():
    parser = argparse.ArgumentParser(description="Generate GPT-informed consent conversations.")

    parser.add_argument("--profiles", type=str, help="Path to existing patient profiles")
    parser.add_argument("--build_profiles", type=int, default=20000, help="Number of profiles to generate")
    parser.add_argument("--consent_pdf", type=str, required=True, help="Path to the consent form PDF")
    parser.add_argument("--replacement", choices=["yes", "no"], default="yes", help="Replace names/PI info in PDF")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (e.g., gpt-4o)")
    parser.add_argument("--fit_only", action="store_true", help="Only use 'fit for trial' patients")
    parser.add_argument("--num_conversations", type=int, default=10, help="Number of conversations to generate")
    parser.add_argument("--max_turns", type=int, default=4, help="Max number of turns per conversation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for conversations")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save metrics CSVs and HTML report")

    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set. Please export it in your shell.")
    
    client = OpenAI(api_key=api_key)

    # load or build profiles
    profiles = load_or_generate_profiles(args.profiles, args.build_profiles)

    # filter by trial eligibility
    if args.fit_only:
        profiles = [p for p in profiles if p.get("fit_for_trial") == "Fit for Trial"]

    # read and clean consent document
    consent_text = extract_text_without_header_footer_and_replace(
        args.consent_pdf, replacement=(args.replacement == "yes")
    )

    # select only the top N profiles
    profiles = profiles[:args.num_conversations]
    num_episodes = len(profiles)

    os.makedirs(args.output_dir, exist_ok=True)
    # save output_file inside output_dir also
    if not os.path.isabs(args.output_file):
        args.output_file = os.path.join(args.output_dir, args.output_file)

    # generate conversations
    conversations = create_pretraining_dataset(
        num_episodes=num_episodes,
        profiles=profiles,
        consent_info=consent_text,
        output_file=args.output_file,
        client=client,
        model=args.model,
        max_turns_cap=args.max_turns,
    )

    # post-generation analytics

    def _row(conv, idx):
        prof = conv.get("profile", {})
        return {
            "conversation_id": idx,
            "patient_age": prof.get("age"),
            "patient_gender": prof.get("gender"),
            "literacy_level": prof.get("literacy_level"),
            "behavior": prof.get("behavior"),
            "interest_level": prof.get("interest_level"),
            "conversation_length": len(conv.get("history", [])),
            "fit_for_trial": prof.get("fit_for_trial"),
            "patient_unanswered": conv.get("patient_unanswered", False),
        }

    data_df = pd.DataFrame([_row(c, i) for i, c in enumerate(conversations)])
    raw_conversations = conversations 

    print(f"\nLoaded {len(data_df)} conversations")
    print(data_df[[
        "conversation_id", "patient_age", "patient_gender",
        "literacy_level", "behavior", "interest_level", "conversation_length"
    ]].head())

    # summary of loaded data
    if not data_df.empty:
        print(f"Age range: {data_df['patient_age'].min()} to {data_df['patient_age'].max()}")
        print(f"Gender distribution: {data_df['patient_gender'].value_counts(dropna=False).to_dict()}")
        if "literacy_level" in data_df.columns:
            print(f"Literacy level distribution: {data_df['literacy_level'].value_counts(dropna=False).to_dict()}")
        if "behavior" in data_df.columns:
            print(f"Behavior distribution: {data_df['behavior'].value_counts(dropna=False).to_dict()}")
        if "interest_level" in data_df.columns:
            print(f"Interest level distribution: {data_df['interest_level'].value_counts(dropna=False).to_dict()}")

    # build results_df from saved conv['metrics']
    results_rows = []
    for i, conv in enumerate(conversations):
        base = {k: data_df.loc[i, k] for k in data_df.columns if k not in {"conversation_length"}}
        metrics = conv.get("metrics", {}) or {}
    
        # Flatten metric scores into columns
        row = {
            **base,
            **{m: metrics.get(m) for m in METRICS.keys() if m in metrics},
        }
    
        # add Patient_consented (if present)
        if "Patient_consented" in metrics:
            row["Patient_consented"] = bool(metrics["Patient_consented"])
    
        results_rows.append(row)
    
    results_df = pd.DataFrame(results_rows)

    # build evidence_df from conv['metrics_evidence']
    evidence_rows = []
    has_any_evidence = any("metrics_evidence" in conv for conv in conversations)
    
    if has_any_evidence:
        for i, conv in enumerate(conversations):
            ev = conv.get("metrics_evidence", {}) or {}
            row = {"conversation_id": i}
    
            # add evidence columns if present, o.w. fill with None
            for m in METRICS.keys():
                row[f"{m}_evidence"] = ev.get(f"{m}_evidence")
                row[f"{m}_turn_number"] = ev.get(f"{m}_turn_number")
            row["Patient_consented_evidence"] = ev.get("Patient_consented_evidence")
            row["Patient_consented_turn"] = ev.get("Patient_consented_turn")
    
            evidence_rows.append(row)
    
        evidence_df = pd.DataFrame(evidence_rows)
    else:
        # empty evidence_df with expected columns (None)
        cols = ["conversation_id"] + \
               [f"{m}_evidence" for m in METRICS.keys()] + \
               [f"{m}_turn_number" for m in METRICS.keys()] + \
               ["Patient_consented_evidence", "Patient_consented_turn"]
        evidence_df = pd.DataFrame(columns=cols)

    # Prints
    if "Patient_consented" in results_df.columns:
        print("\nUnbiased Patient_consented distribution:")
        print(results_df["Patient_consented"].value_counts(dropna=False))
        print(f"Consent rate: {results_df['Patient_consented'].mean():.1%}")
    
        conversation_ids = results_df["conversation_id"].tolist()
        print("Conversation IDs in current sample:")
        print(f"IDs: {conversation_ids}")
    
        print("\nConsent decisions by conversation:")
        for _, row in results_df.iterrows():
            print(f"Conversation {row['conversation_id']}: {row.get('Patient_consented')}")
    
        consented_ids = results_df[results_df["Patient_consented"] == True]["conversation_id"].tolist()
        declined_ids  = results_df[results_df["Patient_consented"] == False]["conversation_id"].tolist()
        print(f"\nConsented: {consented_ids}")
        print(f"Declined: {declined_ids}")
    
    # average per-metric scores
    metric_cols = [m for m in METRICS.keys() if m in results_df.columns]
    if metric_cols:
        avg_scores = results_df[metric_cols].mean().sort_values(ascending=False)
        print("\nAverage scores for each metric:")
        print(avg_scores)
    
    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    metrics_csv  = os.path.join(args.output_dir, f"conversation_metrics_{ts}.csv")
    evidence_csv = os.path.join(args.output_dir, f"conversation_metrics_evidence_{ts}.csv")
    results_df.to_csv(metrics_csv, index=False)
    evidence_df.to_csv(evidence_csv, index=False)
    print(f"Saved:\n- {metrics_csv}\n- {evidence_csv}")

    # conversation_ids for the HTML report
    conversation_ids = results_df["conversation_id"].tolist()
    
    # manual consent review and save file (HTML)
    try:
        html_path = save_my_analysis(
            conversation_ids=conversation_ids,
            raw_conversations=raw_conversations,  
            results_df=results_df,
            evidence_df=evidence_df,
            output_dir=args.output_dir
        )
        print(f"HTML report saved to: {html_path}")
    except Exception as e:
        print(f"Skipping HTML report: {e}")

    # plots
    plot_distributions(profiles, os.path.join(args.output_dir, "plot_distributions.png"))
    plot_top_feature_ratios_all_features(
        conversations=raw_conversations,
        output_prefix=os.path.join(args.output_dir, "top5"),  # top5_consented.png and top5_no_consent.png
        min_total=3,
        top_k=5
    )
    
    # df_feat = plot_top_feature_ratios_all_features(
    #     conversations=merged_convs,
    #     output_prefix="top5",  # produces top5_consented.png and top5_no_consent.png
    #     min_total=3,
    #     top_k=5
    # )
    
    # # Optional: save the table used for plotting
    # if df_feat is not None:
    #     df_feat.to_csv("feature_value_ratios.csv", index=False)


if __name__ == "__main__":
    main()