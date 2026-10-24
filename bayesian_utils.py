import math

SEVERITY_EVIDENCE = {
    "Critical": {
        "fire": 0.75,
        "smoke": 0.65,
        "trapped": 0.85,
        "unconscious": 0.80,
        "bleeding": 0.70,
        "chest pain": 0.75,
        "gas leakage": 0.70,
        "collapse": 0.75,
        "child": 0.55,
        "elderly": 0.50
    },
    "High": {
        "fire": 0.55,
        "smoke": 0.50,
        "trapped": 0.60,
        "unconscious": 0.55,
        "bleeding": 0.55,
        "accident": 0.55,
        "flood": 0.55,
        "injury": 0.50,
        "child": 0.45,
        "elderly": 0.40
    },
    "Medium": {
        "accident": 0.40,
        "injury": 0.35,
        "flood": 0.35,
        "help": 0.30,
        "emergency": 0.30
    },
    "Low": {
        "help": 0.20,
        "emergency": 0.15,
        "accident": 0.15
    }
}

PRIOR = {
    "Critical": 0.20,
    "High": 0.30,
    "Medium": 0.30,
    "Low": 0.20
}

def bayesian_severity_confidence(text):
    text = str(text).lower()

    scores = {}
    matched_evidence = []

    for severity, evidence in SEVERITY_EVIDENCE.items():
        score = PRIOR[severity]
        found = False

        for keyword, likelihood in evidence.items():
            if keyword in text:
                score *= likelihood
                found = True
                if keyword not in matched_evidence:
                    matched_evidence.append(keyword)

        if not found:
            score *= 0.30

        scores[severity] = score

    total = sum(scores.values())

    if total == 0:
        probabilities = {k: 0.25 for k in scores}
    else:
        probabilities = {k: v / total for k, v in scores.items()}

    final_severity = max(probabilities, key=probabilities.get)

    return {
        "probabilities": probabilities,
        "final_severity": final_severity,
        "confidence": probabilities[final_severity],
        "matched_evidence": matched_evidence
    }