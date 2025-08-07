import argparse
import json
import os
import datetime
from openai import OpenAI
from profiles import generate_patient_profile
from pdf_reader import extract_text_without_header_footer_and_replace
from gpt_conversations import create_pretraining_dataset
from plotting import (
    plot_distributions,
    plot_categorical_distributions,
    plot_delay_by_field,
    plot_top_features,
)


def load_or_generate_profiles(path, build_count):
    if path:
        with open(path) as f:
            return json.load(f)
    else:
        profiles = [generate_patient_profile() for _ in range(build_count)]
        filename = "patient_profiles.json"
        if os.path.exists(filename):
            suffix = datetime.datetime.now().strftime("_%m%d%Y")
            filename = filename.replace(".json", f"{suffix}.json")
        with open(filename, "w") as f:
            json.dump(profiles, f, indent=2)
        return profiles


def main():
    parser = argparse.ArgumentParser(description="Generate GPT-informed consent conversations.")

    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--profiles", type=str, help="Path to existing patient profiles")
    parser.add_argument("--build_profiles", type=int, default=2000, help="Number of profiles to generate")
    parser.add_argument("--consent_pdf", type=str, required=True, help="Path to the consent form PDF")
    parser.add_argument("--replacement", choices=["yes", "no"], default="yes", help="Replace names/PI info in PDF")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (e.g., gpt-4o)")
    parser.add_argument("--fit_only", action="store_true", help="Only use 'fit for trial' patients")
    parser.add_argument("--num_conversations", type=int, default=10, help="Number of conversations to generate")
    parser.add_argument("--max_turns", type=int, default=8, help="Max number of turns per conversation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for conversations")

    args = parser.parse_args()
    client = OpenAI(api_key=args.api_key)

    # Load or build profiles
    profiles = load_or_generate_profiles(args.profiles, args.build_profiles)

    # Filter by trial eligibility
    if args.fit_only:
        profiles = [p for p in profiles if p.get("fit_for_trial") == "Fit for Trial"]

    # Read and sanitize consent document
    consent_text = extract_text_without_header_footer_and_replace(
        args.consent_pdf, replacement=(args.replacement == "yes")
    )

    # Select only the top N profiles
    profiles = profiles[:args.num_conversations]

    # Generate conversations
    conversations = create_pretraining_dataset(
        num_episodes=args.num_conversations,
        profiles=profiles,
        consent_info=consent_text,
        output_file=args.output_file,
        client=client,
        model=args.model,
        max_turns=args.max_turns,
    )

    # Print statistics
    stats = {"the patient consented": 0, "the patient needs more time": 0, "the patient did not give consent": 0}
    for c in conversations:
        decision = c.get("consent_decision", "unknown")
        if decision in stats:
            stats[decision] += 1

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Generate plots
    plot_distributions(profiles, "plot_distributions.png")
    plot_categorical_distributions(profiles, "plot_categorical_distributions.png")
    plot_delay_by_field(conversations, field="behavior", output_path="delay_by_behavior.png")
    plot_top_features(conversations, "the patient consented", "literacy_level", "top_features_consented.png", "Top features that consented")
    plot_top_features(conversations, "the patient needs more time", "literacy_level", "top_features_more_time.png", "Top features needing more time")
    plot_top_features(conversations, "the patient did not give consent", "literacy_level", "top_features_no_consent.png", "Top features most associated with no consent")


if __name__ == "__main__":
    main()