disease_db = {
    "fever": {"disease": "Foot-and-mouth disease", "treatment": "Isolate infected cows and use antiseptic ointments."},
    "cough": {"disease": "Bovine Respiratory Disease", "treatment": "Administer antibiotics and ensure clean ventilation."},
    "diarrhea": {"disease": "Johne’s disease", "treatment": "Provide hydration therapy and consult a veterinarian."},
    "lesions": {"disease": "Lumpy Skin Disease", "treatment": "Use antiviral drugs and supportive hydration care."}
}

def diagnose_disease_with_treatment(symptoms):
    # Split symptoms into a list if they are comma-separated
    symptom_list = [sym.strip().lower() for sym in symptoms.split(",")]
    
    matched = []
    for symptom in symptom_list:
        if symptom in disease_db:
            matched.append({**{"symptom": symptom}, **disease_db[symptom]})
    
    return matched if matched else [{"disease": "No known disease", "treatment": "Consult a veterinarian."}]