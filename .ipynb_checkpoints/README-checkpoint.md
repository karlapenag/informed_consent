# Informed consent simulation

This pipeline generates **simulated patient-clinical AI assistant conversations** for informed consent based on structured patient profiles and a consent PDF using GPT models.  

---

## Folder structure

```
.
├── main.py
├── profiles.py
├── pdf_reader.py
├── gpt_conversations.py
├── plotting.py
├── evaluator_function_edited.py
├── environment.yml
├── patient_profiles.json               # saved/generated profile file
├── [output_dir]
│   ├── [output_file].json              # GPT generated conversations          
│   ├── plots.png                       # Output plots (3)
│   ├── metrics.csv                     # CSV metric output
│   ├── metrics_evidence.csv            # CSV metric evidence output
│   ├── consent_analysis.html           # html file
└── README.md
```

---

## Requirements

You need **Conda** installed. You can install either:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
- [Anaconda](https://www.anaconda.com/)

### Required files (inputs)
1. **Consent form PDF**  
   You must provide a PDF file containing the informed consent documentation (sample in consent_example dir).  
   Example: `my_consent_form.pdf`

2. **OpenAI API Key**  
   You’ll need an OpenAI API key with access to GPT models (e.g. `gpt-4o`, `gpt-3.5-turbo`).  
   You can create one at: https://platform.openai.com/account/api-keys
   Make sure to set your OpenAI key before running:
   ```bash
    export OPENAI_API_KEY="your-key"
    echo $OPENAI_API_KEY
   ```

---

## Setup

### 1. Clone the repository (or place all `.py` files in one folder)

```bash
cd /path/to/your/folder
```

### 2. Create environment

```bash
conda env create -f environment.yml
conda activate generate_convs_env
```

For Jupyter notebook, you can also do:

```bash
conda install ipykernel
python -m ipykernel install --user --name generate_convs_env
```

---

## How to run

From the terminal, **inside the project folder**, example run:

```bash
python main.py \
  --consent_pdf consent_example/Consent_PROMISE.pdf \
  --output_file generated_convs.json \
  --fit_only \
  --num_conversations 5
```

### Optional arguments

| Argument                  | Type     | Default        | Description |
|--------------------------|----------|----------------|-------------|
| `--consent_pdf`          | path     | _Required_     | Path to consent PDF |
| `--output_file`          | path     | _Required_     | Output JSON file for conversations |
| `--profiles`             | path     | `None`         | Load existing patient profile file |
| `--build_profiles`       | int      | `20000`        | Number of profiles to generate |
| `--replacement`          | yes/no   | `yes`          | If `no`, skip replacing names in PDF |
| `--model`                | string   | `gpt-4o`       | GPT model to use |
| `--fit_only`             | flag     | `False`        | Use only profiles with `fit_for_trial = True` |
| `--num_conversations`    | int      | `10`           | Number of conversations to generate |
| `--max_turns`            | int      | `4`            | Max number of turns per conversation |
| `--output_dir`           | path     | `output`           | Directory to save plots, CSVs, and reports |

---

## Outputs

After execution, the following will be created in --output_dir (default: output):

### JSON output

| File                        | Description |
|----------------------------|-------------|
| `conversations_output.json`| List of conversation dictionaries with history, decision, and profile info |

Each entry in this file includes:
- `profile`: the patient's data
- `history`: alternating GPT turns
- `metrics`: score for each informed consent metric
- `consent_decision`: `the patient consented`, or `the patient did not consent`
- `patient_unanswered`: whether patient left questions unanswered

---

### CSV Metric Files
- `conversation_metrics_YYYYMMDD_HHMM.csv` – Scores per conversation
- `conversation_metrics_evidence_YYYYMMDD_HHMM.csv` – Metric explanations

---

### Plots (.png)

| File                          | Description |
|------------------------------|-------------|
| `plot_distributions.png`     | Subplot of distributions for profile fields |
| `top_features_consented.png` | Top 5 features that consented |
| `top_features_no_consent.png`| Top 5 features most associated with no consent |

---

### HTML Report

- `consent_analysis_YYYYMMDD_HHMM.html` – Interactive analysis summary
  
---

## Notes

- **PDF replacements:** By default, placeholder fields (names, phones, institution, etc.) are replaced with defaults. You can pass a custom replacement dictionary in future versions for other protocols.
- **Profiles file:** If none is given, a new one will be created. If `patient_profiles.json` already exists, it adds a suffix like `_08252025` (date of creation).
- **Consent PDF parsing:** The script removes headers/footers and replaces names/phones/institution placeholders (unless `--replacement no` is set).
- **Conversation simulation:** Each conversation alternates between a GPT-based assistant and a GPT-simulated patient based on personality traits and medical data.
- **API calls:** GPT is called multiple times per conversation. 10 conversations may require ~100 calls.

---

## 

If you run into issues with OpenAI rate limits, try:
- Reducing `--num_conversations`
- Using a cheaper model (`--model gpt-3.5-turbo`)
