import streamlit as st
import pandas as pd
import numpy as np
import joblib
import whisper

st.set_page_config(
    page_title="RescueNet",
    page_icon="🚨",
    layout="wide"
)

# ------------------------------
# Load trained ML models
# ------------------------------
vectorizer = joblib.load("models/vectorizer.pkl")
type_model = joblib.load("models/emergency_type_model.pkl")
severity_model = joblib.load("models/severity_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")
resource_model = joblib.load("models/resource_model.pkl")

resource_names = ["Ambulance", "Fire Brigade", "Police", "Rescue Team"]

# ------------------------------
# Initialize session state
# ------------------------------
if "emergency_text" not in st.session_state:
    st.session_state.emergency_text = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "result" not in st.session_state:
    st.session_state.result = {}

st.title("🚨 RescueNet")
st.subheader("AI-Based Emergency Reporting and Resource Allocation System")

tab1, tab2 = st.tabs(["🆘 Report Emergency", "🎛️ Control Room"])

# ------------------------------
# Citizen SOS Page
# ------------------------------
with tab1:
    st.header("Report an Emergency")

    input_method = st.radio(
        "Choose input method",
        ["🎤 Voice Input", "⌨️ Text Input"]
    )

    if input_method == "🎤 Voice Input":
        audio = st.audio_input("Press the button and speak your emergency")

        if audio is not None:
            with open("emergency_audio.wav", "wb") as f:
                f.write(audio.getbuffer())

            st.audio("emergency_audio.wav", format="audio/wav")

            if st.button("Convert Voice to Text and Analyze"):
                with st.spinner("Listening and analyzing emergency..."):
                    whisper_model = whisper.load_model("base")
                    result = whisper_model.transcribe("emergency_audio.wav")

                    st.session_state.emergency_text = result["text"]

                    features = vectorizer.transform([st.session_state.emergency_text])

                    emergency_type = type_model.predict(features)[0]
                    severity = severity_model.predict(features)[0]
                    priority = priority_model.predict(features)[0]

                    resources = resource_model.predict(features)
                    resource_values = np.array(resources).reshape(-1)

                    required_resources = [
                        name for name, value in zip(resource_names, resource_values)
                        if value == 1
                    ]

                    st.session_state.result = {
                        "emergency_type": emergency_type,
                        "severity": severity,
                        "priority": priority,
                        "required_resources": required_resources
                    }

                    st.session_state.analysis_done = True

    else:
        text_input = st.text_area(
            "Describe the emergency",
            value=st.session_state.emergency_text,
            placeholder="Example: Fire in a building, people trapped inside"
        )

        if st.button("Analyze Emergency") and text_input:
            st.session_state.emergency_text = text_input

            features = vectorizer.transform([text_input])

            emergency_type = type_model.predict(features)[0]
            severity = severity_model.predict(features)[0]
            priority = priority_model.predict(features)[0]

            resources = resource_model.predict(features)
            resource_values = np.array(resources).reshape(-1)

            required_resources = [
                name for name, value in zip(resource_names, resource_values)
                if value == 1
            ]

            st.session_state.result = {
                "emergency_type": emergency_type,
                "severity": severity,
                "priority": priority,
                "required_resources": required_resources
            }

            st.session_state.analysis_done = True

    # --------------------------
    # Show saved result
    # --------------------------
    if st.session_state.analysis_done:
        st.success("Voice converted successfully")
        st.write("**Transcribed Emergency Report:**")
        st.info(st.session_state.emergency_text)

        result = st.session_state.result

        st.subheader("AI Emergency Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Emergency Type", result["emergency_type"])
            st.metric("Severity", result["severity"])

        with col2:
            st.metric("Priority", f"Priority {result['priority']}")
            st.metric(
                "Required Resources",
                ", ".join(result["required_resources"])
                if result["required_resources"] else "None"
            )

        if result["severity"] == "Critical":
            st.error("⚠️ Critical emergency detected. Immediate dispatch recommended.")
        elif result["severity"] == "High":
            st.warning("⚠️ High-priority emergency detected.")
        else:
            st.success("Emergency recorded successfully.")

# ------------------------------
# Control Room Page
# ------------------------------
with tab2:
    st.header("Control Room Dashboard")

    st.info(
        "This dashboard shows emergency reports, AI predictions, "
        "priority-based queue, and resource dispatch status."
    )

    st.subheader("Emergency Queue")

    if st.session_state.analysis_done:
        result = st.session_state.result

        queue_data = {
            "Emergency ID": ["E1001"],
            "Emergency Type": [result["emergency_type"]],
            "Severity": [result["severity"]],
            "Priority": [result["priority"]],
            "Required Resources": [
                ", ".join(result["required_resources"])
                if result["required_resources"] else "None"
            ],
            "Status": ["Pending"]
        }
    else:
        queue_data = {
            "Emergency ID": [],
            "Emergency Type": [],
            "Severity": [],
            "Priority": [],
            "Required Resources": [],
            "Status": []
        }

    queue_df = pd.DataFrame(queue_data)
    st.dataframe(queue_df, use_container_width=True)