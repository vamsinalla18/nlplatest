import gradio as gr
import pandas as pd
from collections import defaultdict

from inference.disease_predictor import DiseasePredictor

predictor = DiseasePredictor()

# ── Symptom display names (patient-friendly) ─────────────────────────────────
DISPLAY = {
    "abdominal_pain":      "Abdominal / Stomach Pain",
    "anxiety_feeling":     "Anxiety / Excessive Worry",
    "appetite_loss":       "Loss of Appetite",
    "back_pain":           "Back Pain",
    "belching":            "Burping / Belching",
    "bleeding":            "Bleeding (coughing / stool / urine)",
    "blisters":            "Blisters / Fluid-filled Sores",
    "bloating":            "Bloating / Gassy",
    "blood_in_stool":      "Blood in Stool",
    "blood_in_urine":      "Blood in Urine",
    "blurred_vision":      "Blurred Vision",
    "body_aches":          "Body Aches / All-over Pain",
    "breathing_difficulty":"Difficulty Breathing / Shortness of Breath",
    "burning_urination":   "Burning / Painful Urination",
    "chest_pain":          "Chest Pain",
    "chest_tightness":     "Chest Tightness / Pressure",
    "chills":              "Chills / Shivering",
    "cloudy_urine":        "Cloudy / Murky Urine",
    "cold_extremities":    "Cold Hands and Feet",
    "cold_intolerance":    "Always Feeling Cold / Sensitive to Cold",
    "cold_sweat":          "Cold Sweats / Clammy Skin",
    "confusion":           "Confusion / Brain Fog / Disorientation",
    "constipation":        "Constipation / Difficulty Passing Stool",
    "cough":               "Cough (dry or wet)",
    "dark_urine":          "Dark / Brown Urine",
    "diarrhea":            "Diarrhea / Loose Stools",
    "dizziness":           "Dizziness / Lightheadedness / Vertigo",
    "dry_skin":            "Dry / Cracked Skin",
    "ear_discharge":       "Fluid / Discharge from Ear",
    "ear_pain":            "Ear Pain / Earache",
    "excessive_hunger":    "Excessive Hunger / Always Hungry",
    "excessive_thirst":    "Excessive Thirst / Always Thirsty",
    "eye_redness":         "Red / Bloodshot Eyes",
    "facial_drooping":     "Face Drooping (one side)",
    "fatigue":             "Fatigue / Extreme Tiredness",
    "fever":               "Fever / High Temperature",
    "flank_pain":          "Side / Flank Pain (near kidney)",
    "frequent_urination":  "Frequent Urination",
    "hair_loss":           "Hair Loss / Thinning Hair",
    "headache":            "Headache",
    "heartburn":           "Heartburn / Acid Reflux",
    "heat_intolerance":    "Always Feeling Hot / Sensitive to Heat",
    "insomnia":            "Insomnia / Trouble Sleeping",
    "irregular_heartbeat": "Irregular Heartbeat / Arrhythmia",
    "itching":             "Itching / Itchy Skin",
    "jaw_pain":            "Jaw Pain",
    "joint_pain":          "Joint Pain",
    "joint_stiffness":     "Joint Stiffness",
    "left_arm_pain":       "Left Arm Pain / Numbness",
    "leg_cramps":          "Leg Cramps",
    "loss_of_balance":     "Loss of Balance / Unsteadiness",
    "loss_of_smell":       "Loss of Smell",
    "loss_of_taste":       "Loss of Taste",
    "mucus_production":    "Phlegm / Mucus / Productive Cough",
    "muscle_aches":        "Muscle Aches / Soreness",
    "muscle_stiffness":    "Muscle Stiffness / Rigidity",
    "nail_changes":        "Nail Changes (pitting, thickening, yellowing)",
    "nasal_congestion":    "Nasal Congestion / Stuffy Nose",
    "nausea":              "Nausea / Feeling Sick",
    "neck_pain":           "Neck Pain / Stiff Neck",
    "nodules":             "Lumps / Nodules on Skin",
    "numbness":            "Numbness / Tingling / Pins and Needles",
    "pain":                "General Pain / Ache",
    "pallor":              "Pale Skin / Looking Pale",
    "pelvic_pain":         "Pelvic / Lower Abdominal Pain",
    "phonophobia":         "Sensitivity to Sound / Noise",
    "photophobia":         "Sensitivity to Light",
    "pus":                 "Pus / Discharge from Wound",
    "rapid_heartbeat":     "Rapid Heartbeat / Heart Racing / Palpitations",
    "rash":                "Skin Rash / Hives / Red Spots",
    "retro_orbital_pain":  "Pain Behind the Eyes",
    "runny_nose":          "Runny Nose",
    "sadness":             "Persistent Sadness / Low Mood / Depression",
    "skin_discoloration":  "Skin Discoloration / Unusual Patches",
    "skin_scaling":        "Scaly / Flaky Skin",
    "slurred_speech":      "Slurred Speech / Difficulty Speaking",
    "slow_healing":        "Slow Healing Wounds / Cuts",
    "sneezing":            "Sneezing",
    "sore_throat":         "Sore Throat",
    "sour_taste":          "Sour / Acid Taste in Mouth",
    "sweating":            "Excessive Sweating / Night Sweats",
    "swelling":            "Swelling (ankles, legs, face, body)",
    "throat_swelling":     "Swollen Throat / Difficulty Swallowing",
    "tremor":              "Tremors / Shaking Hands / Twitching",
    "vision_loss":         "Sudden Vision Loss / Blindness",
    "visible_veins":       "Visible / Bulging Veins",
    "vomiting":            "Vomiting / Throwing Up",
    "watery_eyes":         "Watery / Itchy Eyes / Tearing",
    "weakness":            "Weakness / Loss of Strength",
    "weight_gain":         "Unexplained Weight Gain",
    "weight_loss":         "Unexplained Weight Loss",
    "wheezing":            "Wheezing / Whistling when Breathing",
    "yellow_skin":         "Yellow Skin / Eyes (Jaundice)",
}

# Reverse: display name → canonical
CANONICAL = {v: k for k, v in DISPLAY.items()}

# Sorted choices for the dropdown
types_df = pd.read_csv("data/entity_types.csv", header=None)
ALL_SYMPTOMS = sorted(e for e, t in zip(types_df[0], types_df[1]) if t == "Symptom")
CHOICES = sorted(DISPLAY.get(s, s.replace("_", " ").title()) for s in ALL_SYMPTOMS)

# ── Build symptom co-occurrence from KG triples ──────────────────────────────
triples_df = pd.read_csv("data/triples.csv", header=None)
disease_symptoms = defaultdict(set)
for _, row in triples_df.iterrows():
    disease_symptoms[row[0]].add(row[2])

cooccurrence = defaultdict(set)
for symptoms in disease_symptoms.values():
    for s1 in symptoms:
        for s2 in symptoms:
            if s1 != s2:
                cooccurrence[s1].add(s2)


# ── Callbacks ────────────────────────────────────────────────────────────────
def update_related(selected_display):
    if not selected_display:
        return gr.CheckboxGroup(choices=[], value=[], visible=False)

    selected_canonical = [CANONICAL.get(d) for d in selected_display if d in CANONICAL]

    related = set()
    for sym in selected_canonical:
        related.update(cooccurrence.get(sym, set()))
    related -= set(selected_canonical)

    related_display = sorted(
        DISPLAY.get(s, s.replace("_", " ").title()) for s in related
    )
    return gr.CheckboxGroup(
        choices=related_display,
        value=[],
        label="Do any of these related symptoms also apply? (select all that fit)",
        visible=bool(related_display),
    )


def predict(selected_display, also_display):
    all_display = list(set((selected_display or []) + (also_display or [])))
    if not all_display:
        return "Please select at least one symptom."

    canonical = [CANONICAL.get(d) for d in all_display if d in CANONICAL]
    results = predictor.predict_from_list(canonical, top_k=5)

    if not results:
        return "No matching diseases found. Try selecting more symptoms."

    lines = ["**Top predictions based on your symptoms:**\n"]
    for i, (disease, score) in enumerate(results, 1):
        bar = "█" * int(score * 40)
        lines.append(f"{i}. {disease.title():<35} {score:.1%}  {bar}")

    return "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Medical Disease Predictor") as demo:

    gr.Markdown(
        """
        # 🏥 Medical Disease Predictor
        Select the symptoms you are experiencing. The system will suggest related symptoms
        and predict the most likely conditions.
        > **Disclaimer:** This tool is for informational purposes only. Always consult a doctor.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            symptom_select = gr.Dropdown(
                choices=CHOICES,
                multiselect=True,
                label="Select Your Symptoms",
                info="Type to search — select everything you are experiencing",
                elem_id="symptom-select",
            )

            related_box = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Do any of these related symptoms also apply?",
                visible=False,
            )

            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")

        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Prediction Results",
                lines=12,
                interactive=False,
                placeholder="Results will appear here after clicking Predict...",
            )

    symptom_select.change(
        fn=update_related,
        inputs=[symptom_select],
        outputs=[related_box],
    )

    predict_btn.click(
        fn=predict,
        inputs=[symptom_select, related_box],
        outputs=[output],
    )

demo.launch(share=True, theme=gr.themes.Soft())
