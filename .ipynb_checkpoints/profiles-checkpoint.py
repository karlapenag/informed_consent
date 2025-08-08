import random
import numpy as np
from scipy.stats import norm, uniform

# Currently implements profile generation using PROMISE protocol (consent_example) criteria.
# Should be modified to automatically adapt to different protocols in future use.

def sample_age(gender):
    if gender == "Male":
        return random.choice(range(30, 90))
    else:
        return random.choice(range(35, 90))

def sample_serum_creatinine():
    return np.clip(norm.rvs(loc=1.0, scale=0.2), 0.4, 2.2)

def sample_BMI():
    return np.clip(norm.rvs(loc=24, scale=4), 14, 50)

def sample_heart_rate():
    return np.clip(norm.rvs(loc=75, scale=10), 50, 120)

def sample_Ankle_Brachial_Index():
    return np.clip(uniform.rvs(loc=1.15, scale=0.25), 0.4, 2.0)

def sample_Agatston_score():
    return int(np.clip(norm.rvs(loc=100, scale=200), 0, 1000))

def check_trial_eligibility(profile):
    # inclusion criteria
    if profile["gender"] == "Male":
        if profile["age"] < 45:
            return "Not Fit for Trial"
        elif 45 <= profile["age"] <= 54:
            if not any([
                profile["risk_factors"]["diabetes"] >= 126,  # Diabetes mellitus
                profile["risk_factors"]["PAD"] >= 50,       # Peripheral arterial disease
                profile["risk_factors"]["cerebrovascular_disease"] >= 50,  # Cerebrovascular disease
                profile["risk_factors"]["tobacco_use"] > 0,  # Ongoing tobacco use
                profile["risk_factors"]["hypertension"][0] > 130 or profile["risk_factors"]["hypertension"][1] > 80,  # Hypertension
                profile["risk_factors"]["ABI"] < 0.9,       # Abnormal ABI
                profile["risk_factors"]["dyslipidemia"] >= 200  # Dyslipidemia
            ]):
                return "Not Fit for Trial"
        elif profile["age"] >= 55:
            pass  # Meets age criteria for males

    elif profile["gender"] == "Female":
        if profile["age"] < 50:
            return "Not Fit for Trial"
        elif 50 <= profile["age"] <= 64:
            if not any([
                profile["risk_factors"]["diabetes"] >= 126,  # Diabetes mellitus
                profile["risk_factors"]["PAD"] >= 50,       # Peripheral arterial disease
                profile["risk_factors"]["cerebrovascular_disease"] >= 50,  # Cerebrovascular disease
                profile["risk_factors"]["tobacco_use"] > 0,  # Ongoing tobacco use
                profile["risk_factors"]["hypertension"][0] > 130 or profile["risk_factors"]["hypertension"][1] > 80,  # Hypertension
                profile["risk_factors"]["ABI"] < 0.9,       # Abnormal ABI
                profile["risk_factors"]["dyslipidemia"] >= 200  # Dyslipidemia
            ]):
                return "Not Fit for Trial"
        elif profile["age"] >= 65:
            pass  # Meets age criteria for females

    # additional inclusion criteria
    if profile["chest_pain_symptoms"] != "new/worsening chest pain":
        return "Not Fit for Trial"
    if profile["serum_creatinine"] > 1.5:
        return "Not Fit for Trial"
    if profile["gender"] == "Female" and profile.get("pregnancy_test") == "Positive":
        return "Not Fit for Trial"
    if not profile.get("planned_test_diagnosis", True):
        return "Not Fit for Trial"

    # exclusion Criteria
    exclusion_flags = profile["exclusion_conditions"]
    if any([
        exclusion_flags["ACS"],
        exclusion_flags["unstable_hemodynamics"],
        exclusion_flags["known_CAD"],
        exclusion_flags["recent_testing"],
        exclusion_flags["significant_conditions"],
        exclusion_flags["CTA_contraindications"],
        exclusion_flags["low_life_expectancy"],
        exclusion_flags["unable_to_consent"],
        exclusion_flags["beta_blocker_ineligible"],
        exclusion_flags["agatston_score_high"] > 800,
        exclusion_flags["BMI_high"] > 40,
        exclusion_flags["cardiac_arrhythmia"] != "None"
    ]):
        return "Not Fit for Trial"

    # If all checks pass, the patient is fit for the trial
    return "Fit for Trial"

def generate_patient_profile():
    profile = {
        "age": None,
        "gender": None,
        "chest_pain_symptoms": "new/worsening chest pain",
        "planned_test_diagnosis": False,
        "risk_factors": {
            "diabetes": None,
            "PAD": None,
            "cerebrovascular_disease": None,
            "tobacco_use": None,
            "hypertension": None,
            "ABI": None,
            "dyslipidemia": None,
        },
        "serum_creatinine": None,
        "pregnancy_test": None,
        "exclusion_conditions": {
            "ACS": False,
            "unstable_hemodynamics": False,
            "known_CAD": False,
            "recent_testing": False,
            "significant_conditions": False,
            "CTA_contraindications": False,
            "low_life_expectancy": False,
            "unable_to_consent": False,
            "beta_blocker_ineligible": False,
            "agatston_score_high": None,
            "BMI_high": None,
            "cardiac_arrhythmia": None,
        },
        "fit_for_trial": None  
    }

    # assign gender and age
    profile["gender"] = random.choice(["Male", "Female"])
    profile["age"] = sample_age(profile["gender"])
    profile["planned_test_diagnosis"] = random.choices([False, True], weights=[0.30, 0.70])[0]

    # serum creatinine level
    profile["serum_creatinine"] = sample_serum_creatinine()

    # pregnancy test for females
    if profile["gender"] == "Female":
        profile["pregnancy_test"] = random.choice(["Positive", "Negative"])

    # risk factors with realistic values
    profile["risk_factors"]["tobacco_use"] = np.clip(norm.rvs(loc=10, scale=15), 0, 80)
    profile["risk_factors"]["diabetes"] = np.clip(norm.rvs(loc=120, scale=40), 80, 300)
    profile["risk_factors"]["PAD"] = np.clip(norm.rvs(loc=55, scale=15), 0, 100)
    profile["risk_factors"]["cerebrovascular_disease"] = np.clip(norm.rvs(loc=55, scale=15), 0, 100)
    profile["risk_factors"]["hypertension"] = (
        np.clip(norm.rvs(loc=140, scale=15), 90, 200),
        np.clip(norm.rvs(loc=90, scale=10), 60, 120)
    )
    profile["risk_factors"]["ABI"] = sample_Ankle_Brachial_Index()
    profile["risk_factors"]["dyslipidemia"] = np.clip(norm.rvs(loc=130, scale=30), 100, 300)

    # exclusion conditions with realistic bias
    bias = [0.80, 0.20]
    for condition in ["ACS", "unstable_hemodynamics", "known_CAD", "recent_testing",
                      "significant_conditions", "CTA_contraindications", "low_life_expectancy",
                      "unable_to_consent", "beta_blocker_ineligible"]:
        profile["exclusion_conditions"][condition] = random.choices([False, True], weights=bias)[0]

    profile["exclusion_conditions"]["agatston_score_high"] = sample_Agatston_score()
    profile["exclusion_conditions"]["BMI_high"] = sample_BMI()
    profile["exclusion_conditions"]["cardiac_arrhythmia"] = random.choice(
        ['None', 'Atrial fibrillation', 'Ventricular tachycardia']
    )

    # check if profile is eligible (out of 20,000 only 6.25% tend to be fit_for_trial)
    profile["fit_for_trial"] = check_trial_eligibility(profile)

    return profile
