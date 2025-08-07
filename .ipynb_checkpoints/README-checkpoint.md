# Informed Consent Conversation Simulation Pipeline

This pipeline generates **simulated patient-chatbot conversations** for informed consent based on structured patient profiles and a consent PDF using GPT models.  
It is designed to be fully **reproducible**, **modular**, and **configurable** via CLI.

---

## Folder Structure

```
.
├── main.py
├── profiles.py
├── pdf_reader.py
├── gpt_conversations.py
├── plotting.py
├── environment.yml
├── patient_profiles.json               # saved/generated profile file
├── [output_file].json                  # GPT conversation outputs
├── plot_distributions.png              # Plot 
├── plot_categorical_distributions.png  # Plot
├── delay_by_behavior.png               # Plot
├── top_features_*.png                  # Plots
└── README.md
```

---

## Requirements

Before running this project, ensure you have **Conda** installed on your machine. You can install either:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
- [Anaconda](https://www.anaconda.com/)

Once installed, use Conda to create and activate the required environment as described below.

### Required Files (Inputs)
1. **Consent Form PDF**  
   You must provide a PDF file containing the informed consent documentation.  
   Example: `my_consent_form.pdf`

2. **OpenAI API Key**  
   You’ll need an OpenAI API key with access to GPT models (e.g. `gpt-4o`, `gpt-3.5-turbo`).  
   You can create one at: https://platform.openai.com/account/api-keys

---

## Setup Instructions

### 1. Clone the repository (or place all `.py` files in one folder)

```bash
cd /path/to/your/folder
```

### 2. Create environment

```bash
conda env create -f environment.yml
conda activate generate_convs_env
```

For Jupyter Notebook, you can also do:

```bash
conda install ipykernel
python -m ipykernel install --user --name generate_convs_env
```

---

## How to Run

From the terminal, **inside the project folder**, run:

```bash
python main.py \
  --api_key YOUR_OPENAI_API_KEY \
  --consent_pdf my_consent_form.pdf \
  --output_file conversations_output.json
```

### Optional Arguments

| Argument                  | Type     | Default        | Description |
|--------------------------|----------|----------------|-------------|
| `--api_key`              | string   | _Required_     | Your OpenAI key |
| `--consent_pdf`          | path     | _Required_     | Path to consent PDF |
| `--output_file`          | path     | _Required_     | Output JSON file for conversations |
| `--profiles`             | path     | `None`         | Load existing patient profile file |
| `--build_profiles`       | int      | `20000`         | Number of profiles to generate |
| `--replacement`          | yes/no   | `yes`          | If `no`, skip replacing names in PDF |
| `--model`                | string   | `gpt-4o`       | GPT model to use |
| `--fit_only`             | flag     | `False`        | Use only profiles with `fit_for_trial = True` |
| `--num_conversations`    | int      | `10`           | Number of conversations to generate |
| `--max_turns`            | int      | `8`            | Max number of turns per conversation |

---

## Outputs

After execution, the following files will be created in the same directory:

### JSON Output

| File                        | Description |
|----------------------------|-------------|
| `conversations_output.json`| List of conversation dictionaries with history, decision, and profile info |

Each entry in this file includes:
- `profile`: the patient's data
- `history`: alternating GPT turns
- `consent_decision`: one of `the patient consented`, `the patient needs more time`, or `the patient did not give consent`

---

### Plots (.png)

| File                          | Description |
|------------------------------|-------------|
| `plot_distributions.png`     | Subplot of distributions for profile fields |
| `plot_categorical_distributions.png` | Subplot of categorical fields |
| `delay_by_behavior.png`      | Delay rate by behavior |
| `top_features_consented.png` | Top 5 features that consented |
| `top_features_more_time.png` | Top 5 features needing more time |
| `top_features_no_consent.png`| Top 5 features most associated with refusal |

---

## Notes

- **Profiles file:** If none is given, a new one will be created. If `patient_profiles.json` already exists, it adds a suffix like `_08252025` (date of creation).
- **Consent PDF parsing:** The script removes headers/footers and replaces names/phones/institution placeholders (unless `--replacement no` is set).
- **Conversation simulation:** Each conversation alternates between a GPT-based assistant and a GPT-simulated patient based on personality traits and medical data.
- **API calls:** GPT is called multiple times per conversation. 10 conversations may require ~100 calls.

---

## 

If you run into issues with OpenAI rate limits, try:
- Reducing `--num_conversations`
- Using a cheaper model (`--model gpt-3.5-turbo`)
