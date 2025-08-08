import json
import pandas as pd
from typing import List, Dict, Any
#from openai import AzureOpenAI
from openai import OpenAI
import os
import glob
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime
import json, jsonschema
import collections
from pathlib import Path


# ---------------------------------------------------------------------------
#  Metric descriptions – pulled straight from your original prompt
# ---------------------------------------------------------------------------
METRICS = {
    # ----------------------- FUNDAMENTAL -----------------------
    "Study_objective": "How clearly was the purpose and objective of the study explained?",
    "Study_procedures": "How thoroughly were the procedures and tests involved in the study described?",
    "Study_risks": "How well were the risks, potential injuries, and their management addressed?",
    "Compensation_treatment": "How clearly was compensation or treatment for participation or potential injuries explained?",
    "Study_commitment": "How well was the time commitment and duration of participation clarified?",
    "Benefits_incentive": "How thoroughly were the potential benefits and incentives for participation explained?",
    "Voluntariness": "How clearly was it communicated that participation is voluntary and what alternatives exist?",
    "Therapeutic_misconception": "How well did the AI assistant avoid creating the impression that the research is primarily for the patient's benefit?",
    "Withdrawal_ability": "How clearly was the right to withdraw from the study at any time explained?",
    "Point_of_contact": "How clearly was contact information provided for concerns about misconduct, injury, or complaints?",
    # ----------------------- SECONDARY -------------------------
    "Completeness_conciseness": "How complete yet concise was the information compared to a standard informed consent form?",
    "Engagement": "How well did the AI assistant engage with the patient during the consent process? Base this also on how many patient interactions were present in the conversation.",
    "Conversational_relevance": "How relevant and responsive were the AI assistant's answers to the patient's questions?",
    "Language_appropriateness": "How appropriate was the language used (avoiding overly technical terms or jargon)? Also based on use of jargon by participant versus CRC AI agent.",
    "Avoid_attrition": "How well did the AI assistant maintain the patient's interest and avoid information overload?",
    "Participant_time": "How respectful was the AI assistant of the participant's time during the conversation?",
    "Participant_verbosity": "How well did the AI assistant allow the participant to express themselves fully?"
}

FUNDAMENTAL = list(METRICS)[:10]   # first 10 items above

#old data
def load_and_prepare_data(file_path: str):
    """Load and organize conversation data for LLM analysis"""
    with open(file_path, 'r') as f:
        conversations = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(conversations)
    
    # Extract additional details that might be useful
    df['patient_age'] = df.apply(lambda row: row['profile']['age'], axis=1)
    df['patient_gender'] = df.apply(lambda row: row['profile']['gender'], axis=1)
    df['conversation_length'] = df.apply(lambda row: len(row['history']), axis=1)
    
    return df, conversations

# New data: 1 json file per conversation
def load_and_prepare_data(folder: str):
    """
    folder/
      consent_chatbot_conv_001.json
      consent_chatbot_conv_002.json
      …
      consent_chatbot_conv_100.json
    """
    paths = sorted(glob.glob(os.path.join(folder, "consent_chatbot_conv_*.json")))
    conversations = []

    for idx, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            convo = json.load(f)

        # attach metadata so we can trace it later
        convo["conversation_id"] = idx
        convo["source_file"]     = os.path.basename(path)
        conversations.append(convo)

    # -------------------- build dataframe ------------------------------
    df = pd.DataFrame(conversations)

    # convenience columns
    df["patient_age"]        = df["profile"].apply(lambda p: p["age"])
    df["patient_gender"]     = df["profile"].apply(lambda p: p["gender"])
    df["conversation_length"] = df["history"].apply(len)
    df["literacy_level"]     = df["profile"].apply(lambda p: p.get("literacy_level"))
    df["behavior"]           = df["profile"].apply(lambda p: p.get("behavior"))
    df["interest_level"]     = df["profile"].apply(lambda p: p.get("interest_level"))

    return df, conversations

def format_conversation_transcript(conversation):
    lines = [
        f"[TURN {i}] {msg['role'].upper()}: {msg['content']}"
        for i, msg in enumerate(conversation["history"], start=1)
    ]
    return "\n\n".join(lines) + "\n"

def print_conversation_json(raw_conversations, conversation_id):
    """Print the raw JSON for a conversation"""
    try:
        conversation = raw_conversations[conversation_id]
        print(json.dumps(conversation, indent=2))
    except (IndexError, KeyError) as e:
        print(f"Error accessing conversation {conversation_id}: {str(e)}")

def print_conversation_details(raw_conversations, conversation_id):
    """Print detailed information about a specific conversation"""
    try:
        # Try to access by index if conversation_id is an index
        conversation = raw_conversations[conversation_id]
        
        # Print conversation profile
        print("\n===== CONVERSATION PROFILE =====")
        profile = conversation['profile']
        print(f"Age: {profile['age']}")
        print(f"Gender: {profile['gender']}")
        print(f"Fit for trial: {profile['fit_for_trial']}")
        print(f"Max turns: {conversation['max_turns']}")
        
        # Print conversation history
        print("\n===== CONVERSATION TRANSCRIPT =====")
        for i, message in enumerate(conversation['history']):
            role = message['role'].upper()
            content = message['content']
            print(f"\n[TURN {i+1}] {role}:")
            print(content)
            
    except (IndexError, KeyError) as e:
        print(f"Error accessing conversation {conversation_id}: {str(e)}")

def save_conversation_to_file(raw_conversations, conversation_id, filename=None):
    """Save a conversation to a text file for detailed analysis"""
    if filename is None:
        filename = f"output/conversations/conversation_{conversation_id}.txt"

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    try:
        conversation = raw_conversations[conversation_id]
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Write profile
            f.write("===== CONVERSATION PROFILE =====\n")
            profile = conversation['profile']
            f.write(f"Age: {profile['age']}\n")
            f.write(f"Gender: {profile['gender']}\n")
            f.write(f"Fit for trial: {profile['fit_for_trial']}\n")
            f.write(f"Max turns: {conversation['max_turns']}\n\n")
            
            # Write conversation history
            f.write("===== CONVERSATION TRANSCRIPT =====\n")
            for i, message in enumerate(conversation['history']):
                role = message['role'].upper()
                content = message['content']
                f.write(f"\n[TURN {i+1}] {role}:\n")
                f.write(f"{content}\n")
        
        print(f"Conversation saved to {filename}")
        
    except (IndexError, KeyError) as e:
        print(f"Error accessing conversation {conversation_id}: {str(e)}")
    
#### main functions


def build_function_schema() -> List[Dict]:
    """
    Build a single function schema that the model must call.  It contains:
      • every metric (score, evidence, turn_number)
      • Patient_consented (boolean)
    """
    metric_obj = {             # …unchanged …
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "evidence": {"type": "string"},
            "turn_number": {"type": ["integer", "null"]}
        },
        "required": ["score", "evidence", "turn_number"]
    }

    props = {m: metric_obj for m in METRICS}

    # ---------- give Patient_consented its own definition ----------
    props["Patient_consented"] = {
        "type": "boolean",
        "description": (
            "True only when the patient gave explicit consent as defined in the "
            "rubric; otherwise False."
        ),
        "default": False          # <-- the extra nudge helps some models
    }

    props["Patient_consented_evidence"] = {
        "type": "object",
        "properties": {
            "evidence":    {"type": "string"},
            "turn_number": {"type": ["integer", "null"]}
        },
        "required": ["evidence", "turn_number"]
    }

    return [{
        "name": "score_consent_conversation",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(props)     # every key must appear
        }
    }]

### json error catching functions

_SCHEMA = build_function_schema()[0]["parameters"]  # already defined

def _validate_metrics(blob: dict):
    """
    Raises jsonschema.ValidationError if the object doesn't match the
    function schema we gave the model.
    """
    jsonschema.validate(instance=blob, schema=_SCHEMA)
    
def get_conversation_metrics_single_shot(
    client,
    conversation_text: str,
    model_deployment: str = "gpt-4o",
    temperature: float = 0.0
) -> dict:
    """
    Calls the model once; model must return the function's arguments
    (guaranteed well-formed JSON).
    """
    # ---------- system & user messages ----------
    sys = (
        "You are a medical-ethics expert. Analyse the conversation, score every "
        "metric 0-1 with two decimals, supply evidence & turn number, and decide "
        "if explicit patient consent was given.  Return ONLY the arguments for "
        "the function you have been asked to call."
    )

    # Build the rubric bullets dynamically from METRICS so it never drifts
    bullets = "\n".join([
    f"- **{m}**: {d}" for m, d in METRICS.items()
    ] + [
        "- **Patient_consented (boolean)**: True only if explicit consent phrases are present; False otherwise.",
        "- **Patient_consented_evidence**: Quote + turn number that justifies the flag."
    ])
    consent_block = """
## Explicit consent = clear statements like
  • "I'm willing to participate" / "I agree to enroll" / "I'm ready to sign" / "I'm ready for next steps"
## Not consent
  • Refusal ("I decline"), postponement ("I need time to think"),
    scheduling without agreement, or ambiguous interest.
Return `"Patient_consented": true` **only** when explicit consent exists.
"""

    user = (
        "# Clinical Trial Informed Consent Conversation\n\n"
        f"{conversation_text}\n\n"
        "# Evaluation Instructions\n\n"
        "Give a 0-1 score for each metric plus evidence & turn.\n"
        "Metrics:\n"
        f"{bullets}\n\n"
        f"{consent_block}"
    )

    # ---------- call Azure ----------
    schema = build_function_schema()[0]
    resp = client.chat.completions.create(
        model=model_deployment,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        tools=[{"type": "function", "function": schema}],
        tool_choice={"type":"function","function":{"name":"score_consent_conversation"}},
        temperature=temperature,
        max_tokens=1500,
    )

    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        raise RuntimeError("Model did not call the expected tool.")
    tool_call = msg.tool_calls[0]
    metrics_json = json.loads(tool_call.function.arguments)

    # ------------------------------------------------------------------
    #  🔒 Safety belt – ensure the key is present
    # ------------------------------------------------------------------
    #print("RAW:", resp.choices[0].message)
    #print("ARGS:", tool_call.function.arguments)
    required = {"Patient_consented", "Patient_consented_evidence", *METRICS}
    missing = required - metrics_json.keys()
    for k in missing:
        if k == "Patient_consented":
            metrics_json[k] = False
        elif k == "Patient_consented_evidence":
            metrics_json[k] = {"evidence": "", "turn_number": None}
        else:
            metrics_json[k] = {"score": 0.0, "evidence": "", "turn_number": None}

    schema_keys = build_function_schema()[0]["parameters"]["properties"].keys()
    assert set(metrics_json) == set(schema_keys), "Schema mismatch!"

    return metrics_json

# Tracks how many extra calls each model needed
retry_counter = collections.Counter()

def get_metrics_with_retries(
    client,
    conversation_text: str,
    model_deployment_primary: str = "gpt-4.1-nano",
    model_deployment_fallback: str = "gpt-4o",
    max_retries: int = 2,
    **kwargs
):
    """
    Calls the single-shot scorer, validates JSON, and automatically retries
    with the same model (up to max_retries).  If it still fails, tries the
    fallback model once.  Raises on final failure.
    """
    model   = model_deployment_primary
    attempt = 0
    while True:
        try:
            metrics = get_conversation_metrics_single_shot(
                client, conversation_text, model_deployment=model, **kwargs
            )
            _validate_metrics(metrics)
            print(f"[info] {model} succeeded on attempt {attempt+1}")
            return metrics
        except Exception as e:
            retry_counter[model] += 1      # <- count this failed call
            attempt += 1
            print(f"[warn] {model} attempt {attempt} failed: {e}")
            if attempt <= max_retries:
                continue          # retry same model
            if model != model_deployment_fallback:
                print(f"[info] escalating to {model_deployment_fallback}")
                model   = model_deployment_fallback
                attempt = 0       # reset counter for the new model
                continue
            raise RuntimeError(f"All attempts failed → {e}")

def split_scores_and_evidence(metrics_json):
    scores, evidence = {}, {}
    for m in METRICS:
        item = metrics_json[m]
        scores[m] = item["score"]
        evidence[f"{m}_evidence"] = item["evidence"]
        evidence[f"{m}_turn_number"] = item["turn_number"]

    scores["Patient_consented"] = metrics_json["Patient_consented"]
    evidence["Patient_consented_evidence"] = metrics_json["Patient_consented_evidence"]["evidence"]
    evidence["Patient_consented_turn"] = metrics_json["Patient_consented_evidence"]["turn_number"]
    return scores, evidence

def process_conversations_to_dataset(
    data_df, raw_conversations, client, model_deployment,
    sample_size=None
):
    """Single-shot pathway; evidence always available."""
    # Sample a subset if specified
    if sample_size and sample_size < len(data_df):
        selected_indices = data_df.sample(sample_size).index.tolist()
    else:
        selected_indices = data_df.index.tolist()
        
    results, evidence_rows = [], []

    for idx in selected_indices:
        conv = raw_conversations[idx]
        profile = conv["profile"]
        transcript = format_conversation_transcript(conv)

        try:
            metrics_json = get_metrics_with_retries(
                client, transcript, model_deployment_primary=model_deployment
            )

            score_dict, evidence_dict = split_scores_and_evidence(metrics_json)

            # ----------- build rows -----------
            results.append({
                "conversation_id": idx,
                "patient_age": profile["age"],
                "patient_gender": profile["gender"],
                "conversation_length": len(conv["history"]),
                "fit_for_trial": profile["fit_for_trial"],
                "literacy_level": profile.get("literacy_level"),
                "behavior": profile.get("behavior"),
                "interest_level": profile.get("interest_level"),
                **score_dict
            })

            evidence_rows.append({
                "conversation_id": idx, 
                **evidence_dict})
            print(f"Processed conversation {idx}")

        except Exception as e:
            print(f"[ERROR] conversation {idx}: {e}")

    results_df = pd.DataFrame(results)
    evidence_df = pd.DataFrame(evidence_rows)
    return results_df, evidence_df

### save functions

def analyze_conversation_consent_to_string(conv_id, raw_conversations, results_df, evidence_df):
    """Modified version that returns string instead of printing"""
    conversation = raw_conversations[conv_id]
    output = []
    
    # Get LLM decision
    llm_decision = results_df[results_df['conversation_id'] == conv_id]['Patient_consented'].iloc[0]
    
    output.append("=" * 60)
    output.append(f"CONVERSATION {conv_id} - LLM Decision: {llm_decision}")
    output.append("=" * 60)
    
    # Show patient profile
    profile = conversation["profile"]
    output.append(
        f"Patient: {profile['age']} yr {profile['gender']}, "
        f"Fit: {profile['fit_for_trial']}, "
        f"Literacy: {profile.get('literacy_level','?')}, "
        f"Behavior: {profile.get('behavior','?')}, "
        f"Interest: {profile.get('interest_level','?')}"
    )
    
    # Get all patient messages
    patient_messages = [msg for msg in conversation['history'] if msg['role'] == 'patient']
    output.append(f"Patient spoke {len(patient_messages)} times")
    
    # ---------- LLM-supplied evidence for its decision ----------
    ev_row = evidence_df[evidence_df["conversation_id"] == conv_id]
    llm_quote = ""
    llm_turn  = None
    if not ev_row.empty:
        llm_quote = ev_row.iloc[0]["Patient_consented_evidence"]
        llm_turn  = ev_row.iloc[0]["Patient_consented_turn"]

    output.append("\nLLM-supplied consent evidence:")
    output.append(f'  Quote : "{llm_quote}"')
    output.append(f"  Turn  : {llm_turn}")
    
    # Show last 2 patient messages (most relevant for consent)
    output.append("\nLAST 2 PATIENT MESSAGES:")
    for i, msg in enumerate(patient_messages[-2:], len(patient_messages)-1):
        output.append(f"\n[PATIENT MSG {i}]:")
        output.append(msg['content'])
    
    # Look for key consent/decline phrases in final message
    if patient_messages:
        final_msg = patient_messages[-1]['content'].lower()
        consent_phrases = ['willing', 'ready', 'agree', 'consent', 'participate', 'join', 'sign', 'enroll', 'proceed']
        decline_phrases = ['decline', 'not interested', 'no thank', 'not ready', 'need time', 'think about', 'consider', 'leaning', 'take some time', 'take the time']
        
        found_consent = [phrase for phrase in consent_phrases if phrase in final_msg]
        found_decline = [phrase for phrase in decline_phrases if phrase in final_msg]
        
        output.append(f"\nKEY PHRASES FOUND:")
        output.append(f"Consent indicators: {found_consent}")
        output.append(f"Decline indicators: {found_decline}")
        
        # Manual assessment
        if found_consent and not found_decline:
            manual_assessment = "LIKELY CONSENTED"
        elif found_decline and not found_consent:
            manual_assessment = "LIKELY DECLINED"
        elif found_consent and found_decline:
            manual_assessment = "AMBIGUOUS"
        else:
            manual_assessment = "UNCLEAR"
            
        output.append(f"Manual assessment: {manual_assessment}")
        match_status = '✅ MATCH' if (llm_decision and 'CONSENT' in manual_assessment) or (not llm_decision and ('DECLINE' in manual_assessment or manual_assessment == 'UNCLEAR')) else '❌ MISMATCH'
        output.append(f"LLM vs Manual: {match_status}")
    
    return "\n".join(output)

def save_analysis_to_html(
    conversation_ids,
    raw_conversations,
    results_df,
    evidence_df,
    filename: str = "conversation_analysis.html",
):
    """Render the HTML report with summary + per-conversation details."""

    # ---------- overall consent stats ----------
    total_count   = len(conversation_ids)
    consent_count = int(results_df["Patient_consented"].sum())
    consent_rate  = consent_count / total_count if total_count else 0.0

    # ---------- compute “match” between LLM and keyword heuristic ----------
    def _llm_manual_match(cid) -> bool:
        """Return True when per-conversation logic would label ✅ MATCH."""
        conv         = raw_conversations[cid]
        llm_decision = bool(
            results_df.loc[results_df["conversation_id"] == cid,
                        "Patient_consented"].iloc[0]
        )

        # --- recreate the manual assessment exactly as in the block view ---
        pt_msgs = [m["content"].lower() for m in conv["history"]
                if m["role"] == "patient"]
        if not pt_msgs:
            manual = "UNCLEAR"
        else:
            final = pt_msgs[-1]
            consent_words  = ["willing","ready","agree","consent","participate",
                            "join","sign","enroll","proceed"]
            decline_words  = ["decline","not interested","no thank","not ready",
                       "need time","think about","consider", "leaning", 
                       "take some time", "take the time"]
            has_yes = any(w in final for w in consent_words)
            has_no  = any(w in final for w in decline_words)

            if has_yes and not has_no:
                manual = "LIKELY CONSENTED"
            elif has_no and not has_yes:
                manual = "LIKELY DECLINED"
            elif has_yes and has_no:
                manual = "AMBIGUOUS"
            else:
                manual = "UNCLEAR"

        # ---- apply the same ‘✅ MATCH’ rule you use in the text block ----
        if llm_decision:
            return "CONSENT" in manual          # only the ‘consented’ path matches
        else:
            return manual in {"LIKELY DECLINED", "UNCLEAR"}  # every other → match


    match_count = sum(_llm_manual_match(cid) for cid in conversation_ids)
    match_rate  = match_count / total_count if total_count else 0.0

    # ---------- HTML skeleton ----------
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Conversation Analysis</title>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; margin:20px; line-height:1.4;
           background:#f5f5f5; }}
    .container {{ max-width:1200px; margin:0 auto; background:#fff; padding:20px;
                  border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,.1); }}
    .conversation {{ border:2px solid #ddd; margin:20px 0; padding:15px;
                     border-radius:8px; background:#fafafa; }}
    .match {{ color:#2e7d32; font-weight:bold; }}
    .mismatch {{ color:#c62828; font-weight:bold; }}
    .summary   {{ background:#e8f5e8; padding:20px; border-radius:8px;
                  margin-bottom:30px; }}
    .stats {{ display:flex; justify-content:space-around; text-align:center; }}
    .stat-item {{ background:#fff; padding:15px; border-radius:8px;
                  box-shadow:0 1px 3px rgba(0,0,0,.1); }}
    .stat-number {{ font-size:24px; font-weight:bold; color:#1565c0; }}
    pre {{ white-space:pre-wrap; font-size:12px;
           font-family:'Courier New', monospace; }}
    h1 {{ color:#1565c0; text-align:center; border-bottom:3px solid #1565c0;
         padding-bottom:10px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Clinical Trial Informed Consent Analysis</h1>
    <p style="text-align:center; color:#666;">
       Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>

    <div class="summary">
      <h2>Summary Statistics</h2>
      <div class="stats">
        <div class="stat-item">
          <div class="stat-number">{total_count}</div>
          <div>Total Conversations</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{consent_count}</div>
          <div>Patients Consented</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{consent_rate:.1%}</div>
          <div>Consent Rate</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{match_count}</div>
          <div>LLM = Manual</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{match_rate:.1%}</div>
          <div>Match Rate</div>
        </div>
      </div>
    </div>
"""

    # ---------- per-conversation blocks ----------
    for cid in conversation_ids:
        text = analyze_conversation_consent_to_string(
            cid, raw_conversations, results_df, evidence_df
        )
        safe_html = (text.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;")
                         .replace("✅ MATCH", '<span class="match">✅ MATCH</span>')
                         .replace("❌ MISMATCH",
                                  '<span class="mismatch">❌ MISMATCH</span>'))
        html_content += f'<div class="conversation"><pre>{safe_html}</pre></div>'

    html_content += """
  </div>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Analysis saved to {os.path.abspath(filename)}")
    print("📁 Open the file in your browser, then print to PDF (Ctrl+P)")
    return filename

# Simple usage function
def save_my_analysis(conversation_ids, raw_conversations, results_df, evidence_df, output_dir="output"):
    """Quick function to save your current analysis"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"consent_analysis_{timestamp}.html"
    
    # Full path
    full_path = os.path.join(output_dir, filename)
    
    return save_analysis_to_html(conversation_ids, raw_conversations, results_df, evidence_df, full_path)

# Full implementation example
if __name__ == "__main__":
    # Load and prepare data
    #file_path = 'simulated_conversations/generated_consent_conversations.json'
    #data_df, raw_conversations = load_and_prepare_data(file_path)
    
    # New data
    data_df, raw_conversations = load_and_prepare_data("consent_chatbot_150convs_extended_examples")
    print(f"Loaded {len(data_df)} conversations")
    print(data_df[["conversation_id", "patient_age", "patient_gender", "literacy_level", 
                   "behavior", "interest_level", "conversation_length"]].head())

    
    # Print summary of loaded data
    print(f"Loaded {len(data_df)} conversations")
    print(f"Age range: {data_df['patient_age'].min()} to {data_df['patient_age'].max()}")
    print(f"Gender distribution: {data_df['patient_gender'].value_counts().to_dict()}")
    
    print(f"Literacy level distribution: {data_df['literacy_level'].value_counts().to_dict()}")
    print(f"Behavior distribution: {data_df['behavior'].value_counts().to_dict()}")
    print(f"Interest level distribution: {data_df['interest_level'].value_counts().to_dict()}")
    
    # setup OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set. Please export it in your shell.")
    
    client = OpenAI(api_key=api_key)
        
    transcript = format_conversation_transcript(raw_conversations[0])
    #metrics = get_conversation_metrics_single_shot(client, transcript, model_deployment="gpt-4o")
    #metrics = get_conversation_metrics_single_shot(client, transcript, model_deployment="gpt-4.1-nano")
    
    metrics = get_metrics_with_retries(
        client, transcript,
        model_deployment_primary="gpt-4.1-nano",
        model_deployment_fallback="gpt-4o"
    )

    print(json.dumps(metrics, indent=2))
    
    print(transcript)
    
    ####### Metrics
           
    results_df_v8, evidence_df_v8 = process_conversations_to_dataset(
        data_df,
        raw_conversations,
        client,
        model_deployment="gpt-4.1-nano",
        sample_size=None
    )

    print("Extra calls per model:", retry_counter)

    print("Unbiased Patient_consented distribution:")
    print(results_df_v8['Patient_consented'].value_counts())
    print(f"Consent rate: {results_df_v8['Patient_consented'].mean():.1%}")

    # Get the conversation IDs from your current sample
    print("Conversation IDs in current sample:")
    conversation_ids = results_df_v8['conversation_id'].tolist()
    print(f"IDs: {conversation_ids}")

    # Show the consent decisions for each
    print("\nConsent decisions by conversation:")
    for idx, row in results_df_v8.iterrows():
        conv_id = row['conversation_id']
        consented = row['Patient_consented']
        print(f"Conversation {conv_id}: {consented}")

    # Separate into consented vs non-consented
    consented_ids = results_df_v8[results_df_v8['Patient_consented'] == True]['conversation_id'].tolist()
    declined_ids = results_df_v8[results_df_v8['Patient_consented'] == False]['conversation_id'].tolist()

    print(f"\nConsented: {consented_ids}")
    print(f"Declined: {declined_ids}")

    # Manual consent review and save file
    save_my_analysis(conversation_ids, raw_conversations, results_df_v8, evidence_df_v8, "output")
    
    # Display example of the dataset format
    print("\nDataset preview:")
    print(results_df_v8.head())
    
    print("V8 Evidence columns:", evidence_df_v8.columns.tolist())
    print("V8 Evidence shape:", evidence_df_v8.shape)
    print(evidence_df_v8.head())
    
    # Look at some actual evidence examples
    print("\nStudy Objective Evidence:")
    print("Conversation:", evidence_df_v8.loc[0, 'Study_objective_evidence'])
    print("Turn:", evidence_df_v8.loc[0, 'Study_objective_turn_number'])
    #print("Explanation:", evidence_df_v8.loc[0, 'Study_objective_explanation'])
    
    # Print raw JSON for a conversation
    print_conversation_json(raw_conversations, 33)
    print_conversation_details(raw_conversations, 33)
    # Save conversation to a file
    save_conversation_to_file(raw_conversations, 33)
    
    # Show average scores for each metric
    print("\nAverage scores for each metric:")
    # metrics_cols = [col for col in results_df_v8.columns 
    #                if col not in ['conversation_id', 'patient_age', 'patient_gender', 
    #                              'conversation_length', 'fit_for_trial', 'Patient_consented', 'literacy_level', 'behavior', 'interest_level']]
    # print(results_df_v8[metrics_cols].mean().sort_values(ascending=False))
    metrics_cols = list(METRICS.keys())  # the 16 metric names
    avg_scores = results_df_v8[metrics_cols].mean().sort_values(ascending=False)
    print("\nAverage scores for each metric:")
    print(avg_scores)
    
    # Save results to CSV
    results_df_v8.to_csv("output/conversation_metrics_new_data.csv", index=False)
    evidence_df_v8.to_csv("output/conversation_metrics_evidence_new_data.csv", index=False)
