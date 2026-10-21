import joblib

vectorizer = joblib.load("models/vectorizer.pkl")
type_model = joblib.load("models/emergency_type_model.pkl")
severity_model = joblib.load("models/severity_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
resource_model = joblib.load("models/resource_model.pkl")

text = "Fire in a building, heavy smoke, people trapped"

features = vectorizer.transform([text])

print("Emergency Type:", type_model.predict(features)[0])
print("Severity:", severity_model.predict(features)[0])
print("Priority:", priority_model.predict(features)[0])
print("Resources [Ambulance, Fire, Police, Rescue]:", resource_model.predict(features)[0])