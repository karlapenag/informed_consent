import json
import random
import numpy as np
from evaluator_function_edited import (
    format_conversation_transcript,
    get_metrics_with_retries,
    split_scores_and_evidence,
    METRICS,
    FUNDAMENTAL
)

# SYSTEM PROMPTS
PATIENT_SYSTEM_PROMPT = """
You are simulating a patient who needs to provide informed consent for a clinical trial.
Your role is to realistically represent how a patient might respond to questions about informed consent.

You have access to a patient profile that describes who you are, including your demographic and medical information.
You should incorporate this knowledge into your responses. It reflects your age, gender, health conditions, and lab results.
You should reference or react to information from your profile when relevant (e.g., if you're told you're ineligible, surprised by your risk factors, or concerned about test results).
Patient profile:
{patient_profile}

Your characteristics:
- Your health literacy is {literacy_level}
- Background knowledge: {background_knowledge}
- Interest level: {interest_level} — {interest_description}
- Behavioral style: {behavior} — {behavior_description}
- You sometimes ask clarifying questions when you don't understand something
- You may express concerns, fears, or confusion
- You will respond as a real patient would, with appropriate emotional reactions

You may ask questions about topics such as:
- The trial’s **purpose**, **risks**, **benefits**, and **procedures**
- What participation involves, including your **rights** and **responsibilities**
- How the study might affect your **health**, **time**, or **daily life**
- Concerns about **long-term impact**, **privacy**, or **compensation**

Ask **only** the questions that feel most relevant or concerning to you, based on your health conditions, literacy level, interest, and behavioral style.

You need to review the conversation history to maintain context and provide consistent responses.
Throughout the conversation, you should gradually move toward making a decision about consent,
but only after your questions are answered and concerns addressed.

Respond naturally to questions from the informed consent chatbot, representing how a real patient 
with your specified characteristics would likely respond.
"""  
CONSENT_BOT_SYSTEM_PROMPT = """
You are an AI assistant designed to obtain informed consent from patients for medical procedures.
Your goal is to ensure patients truly understand the procedure, risks, benefits, and alternatives.

You have access to the patient’s clinical profile, which includes demographic data (age, gender), risk factors, lab results, medical history, and trial eligibility status. 
Use this information throughout the conversation to tailor your explanations to the patient’s specific health context.

Follow these guidelines for obtaining proper informed consent:
1. Thoroughly review the Consent Information provided to you.
2. Review the conversation history to maintain context and provide consistent responses.
3. Limit the number of questions to no more than 2 per response to simulate a natural conversation.
4. Explain the procedure in clear, non-technical language
5. Detail all significant risks and their likelihood
6. Explain expected benefits and likelihood of success
7. Discuss reasonable alternatives to the procedure
8. Clarify that the patient has the right to refuse
9. Verify the patient's understanding by asking them to explain key points
10. Adjust your explanation based on the patient's health literacy level
11. Address any concerns or questions thoroughly
12. Be honest and transparent without causing undue alarm
13. Adjust your explanation style to match the patient’s literacy level and communication style.

CRITICAL: After thorough explanation and addressing concerns (typically by the 3rd or 4th exchange), 
you MUST explicitly ask for the patient's decision with questions like:
- "Based on what we've discussed, do you consent to proceeding with the [procedure]?"
- "Would you like to go ahead with the procedure, or do you need more time to think about it?"
- "Are you ready to give your consent for this procedure, or do you have additional questions?"

If the patient indicates they're "leaning toward" a decision or seems close to deciding:
- Acknowledge their inclination
- Ask them directly if they're ready to make a final decision
- Use phrases like "It sounds like you're considering proceeding with the surgery. Are you ready to give your consent?"

If the patient indicates they need more time to think or discuss with family:
- Respect this decision completely - this is a valid and important patient right
- Do not pressure them or repeatedly ask for consent in subsequent messages
- Offer support for their decision-making process
- Clarify next steps (e.g., "When you're ready to decide, please contact us with your decision or any questions")
- Avoid redundant questions about consent in later messages once they've indicated they need time

Remember that success is not measured by obtaining consent, but by ensuring the patient makes a fully informed 
decision that they're comfortable with, whether that's consent, refusal, or taking more time to decide.

Patient profile:
{patient_profile}

Consent Information: {consent_info}
"""  

# --- PROFILE FORMATTER ---

def strip_non_patient_visible_fields(profile, for_doctor=False):
    """
    removes fields from patients profile
    """
    profile_copy = dict(profile)
    hidden_fields = [
        "literacy_level", "background_knowledge", "interest_level",
        "interest_description", "behavior", "behavior_description"
    ]
    for field in hidden_fields:
        profile_copy.pop(field, None)
        
    if not for_doctor:
        profile_copy.pop("fit_for_trial", None)    # patient doesn't know if they are fit for the trial or not
        
    return profile_copy

def preprocess_for_serialization(data):
    if isinstance(data, dict):
        return {k: preprocess_for_serialization(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_for_serialization(x) for x in data]
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def format_profile_for_prompt(profile, for_doctor=False):
    clean = preprocess_for_serialization(strip_non_patient_visible_fields(profile, for_doctor))
    return json.dumps(clean, indent=2)

# --- PROMPT WRAPPERS ---

def prepare_doctor_context(consent_info, profile):
    return CONSENT_BOT_SYSTEM_PROMPT.format(
        patient_profile=format_profile_for_prompt(profile, for_doctor=True),
        consent_info=consent_info
    )

def prepare_patient_profile(profile):
    return PATIENT_SYSTEM_PROMPT.format(
        patient_profile=format_profile_for_prompt(profile),
        literacy_level=profile["literacy_level"],
        background_knowledge=profile["background_knowledge"],
        interest_level=profile["interest_level"],
        interest_description=profile["interest_description"],
        behavior=profile["behavior"].split("—")[0].strip(),
        behavior_description=profile["behavior_description"]
    )

# --- OPENAI WRAPPER ---

def generate_response(prompt, client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- FULL EPISODE ---

def generate_doctor_patient_episode(profile, consent_info, max_turns, client, model):
    """
    Two-phase flow:
      Phase 1: run normal conversation for `max_turns` assistant-patient exchanges
      Phase 2: remediation loop for fundamental topics with score < 0.4, batched by the
               number of weak topics; re-score after each batch; stop when
               all fundamentals are covered or a safety cap is reached (capped to 3).
      Final:   extract consent decision and return the episode
    """
    history = []
    decision = None
    patient_unanswered_flag = False
    turn = 0 # counts assistant-patient pairs

    doctor_context = prepare_doctor_context(consent_info, profile)
    patient_profile = prepare_patient_profile(profile)
    
    # Phase 1
    while turn < max_turns:
        # AI Assistant Turn
        query = doctor_context + "\n### Conversation History:" + "".join(
            f"\n- {msg['role']}: {msg['content']}" for msg in history
        ) + f"\n\n### Turn {turn+1}. As the AI assistant, provide your response:"
        ai_response = generate_response(query, client, model)
        if ai_response.lower().startswith("ai assistant:"):
            ai_response = ai_response[len("ai assistant:"):].strip()
        history.append({"role": "AI assistant", "content": ai_response})

        # Patient Turn
        query = patient_profile + "\n### Conversation History:" + "".join(
            f"\n- {msg['role']}: {msg['content']}" for msg in history
        ) + f"\n\n### Turn {turn+1}. As the patient, provide your response:"
        patient_response = generate_response(query, client, model)
        if patient_response.lower().startswith("patient:"):
            patient_response = patient_response[len("patient:"):].strip()
        history.append({"role": "patient", "content": patient_response})
        
        turn += 1 # full exchange completed

    # Phase 2
    MAX_REMEDIATION_CYCLES = 3 # avoid infinite remediation cycles
    remediation_cycles = 0
    scores = {}
    
    while remediation_cycles < MAX_REMEDIATION_CYCLES:
        transcript = format_conversation_transcript({"history":history}) # score the current transcript
        metrics_json = get_metrics_with_retries(client, transcript) # could change this function so that it takes optional topics_to_score to reduce token usage and only re-score everything in the last remediation cycle
        scores, evidence = split_scores_and_evidence(metrics_json)

        # check last patient turn for questions
        last_patient_turn = next(
            (msg["content"] for msg in reversed(history) if msg["role"] == "patient"), 
            ""
        )
        question_check_prompt = (
            "The following is the last response of a patient in a conversation with an AI assistant. "
            "Determine if the patient still has unresolved doubts or concerns based on their last response.\n\n"
            f"### Patient last response:\n{last_patient_turn}\n\n"
            "Reply strictly with one of the following options:\n"
            "- the patient does not have more questions\n"
            "- the patient has more questions"
        )
        question_reply = generate_response(question_check_prompt, client, model="gpt-4.1-nano").strip().lower()
        has_more_questions = "has more questions" in question_reply

        # weak fundamentals
        low_fundamentals = [
            topic for topic in FUNDAMENTAL 
            if scores.get(topic, 0.0) < 0.4
        ]
        
        if not low_fundamentals and not has_more_questions:
            # fundamentals covered
            break
            
        if not low_fundamentals and has_more_questions: # unusual case, keep track of times it happened, could add a fallback prompt and extent logic later
            print("Patient was left with more questions (no low fundamentals).")
            patient_unanswered_flag = True
            break
            
        remediation_cycles += 1
        
        n_needed_turns = max(1, len(low_fundamentals)) # add at least len(low_fundamentals) turns
        low_fundamentals_sorted = sorted(low_fundamentals, key=lambda t: scores[t]) # sort them by score (ascending)
        explanations = "\n".join(
            f"- **{topic.replace('_', ' ').capitalize()}**: {METRICS[topic]}" 
            for topic in low_fundamentals_sorted
        )
    
        for _ in range(n_needed_turns):
            # Turn for assistant to address remaining weak topics
            prompt = (
                doctor_context +
                "\n### Conversation History:" +
                "".join(f"\n- {msg['role']}: {msg['content']}" for msg in history) +
                "\n\nThe following key topics may not have been fully addressed yet:\n" +
                explanations +
                (
                    f"\n\n### Turn {turn+1}. As the AI assistant, continue the conversation naturally. Acknowledge what the patient just said, "
                    "then seamlessly explain or reinforce the topics listed above that haven't been clearly addressed yet. "
                    "Avoid repeating content already covered, and adapt your response to the flow and tone of the conversation."
                )
            )
            new_ai_response = generate_response(prompt, client, model)
            history.append({"role": "AI assistant", "content": new_ai_response})
    
            # Add patient response
            patient_query = (
                patient_profile +
                "\n### Conversation History:" +
                "".join(f"\n- {msg['role']}: {msg['content']}" for msg in history) +
                f"\n\n### Turn {turn+1}. As the patient, provide your response:"
            )
            new_patient_response = generate_response(patient_query, client, model)
            history.append({"role": "patient", "content": new_patient_response})

            turn += 1 # full exchange completed
            
        max_turns += n_needed_turns # extend cap just to keep track
        # Loop again: re-score and continue if still weak topics
                   
    # Final consent decision    
    consented_flag = scores['Patient_consented']
    decision = "the patient consented" if consented_flag == 1 else "the patient did not consent"

    conv = {
        "profile": profile,
        "total_turns": max_turns,
        "history": history,
        "metrics": scores,
        "metrics_evidence": evidence,
        "consent_decision": decision,
        "patient_unanswered": patient_unanswered_flag
    }
    return conv

# --- MAIN LOOP ---

def create_pretraining_dataset(num_episodes, profiles, consent_info, output_file, client, model, max_turns_cap=4):
    dataset = []
    for ep in range(num_episodes):
        profile = profiles[ep]
        max_turns = max_turns_cap
        print(f"\nGenerating Conversation {ep + 1}/{num_episodes}")
        episode = generate_doctor_patient_episode(profile, consent_info, max_turns, client, model)
        dataset.append(episode)

    processed = preprocess_for_serialization(dataset)
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=2)

    return dataset