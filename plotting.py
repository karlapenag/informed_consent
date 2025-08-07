import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter


def plot_distributions(profiles, output_path="plot_distributions.png"):
    df = pd.DataFrame(profiles)
    fields = ["literacy_level", "background_knowledge", "interest_level", "behavior"]
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


def plot_categorical_distributions(profiles, output_path="plot_categorical_distributions.png"):
    df = pd.DataFrame(profiles)
    categorical_fields = ["literacy_level", "background_knowledge", "interest_level", "behavior"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, field in zip(axes.flatten(), categorical_fields):
        if field in df.columns:
            sns.countplot(data=df, x=field, ax=ax)
            ax.set_title(f"Distribution of {field}")
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_delay_by_field(conversations, field, output_path):
    df = pd.DataFrame([c["profile"] for c in conversations])
    df["consent"] = [c["consent_decision"] for c in conversations]
    delay_df = df[df["consent"] == "the patient needs more time"]
    delay_rates = delay_df[field].value_counts(normalize=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=delay_rates.index, y=delay_rates.values)
    plt.title(f"Delay Rate by {field}")
    plt.xlabel(field)
    plt.ylabel("Delay Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)


def plot_top_features(conversations, decision_label, field, output_path, title):
    df = pd.DataFrame([c["profile"] for c in conversations])
    df["decision"] = [c["consent_decision"] for c in conversations]

    subset = df[df["decision"] == decision_label]
    top_features = subset[field].value_counts().head(5)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features.index, y=top_features.values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel(field)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)