import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from io import BytesIO

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("Wine Quality Predictor")

st.sidebar.header("Model")

uploaded_file = st.sidebar.file_uploader("Upload model file (optional)", type=["pkl", "pickle", "pk1", "joblib", "sav"]) 
model_path = st.sidebar.text_input("Or use model path", "best_wine_quality.pk1")


@st.cache_data
def load_model_from_file(file_obj):
    """Try joblib then pickle to load a model from a file-like object or path-like object."""
    # joblib.load accepts file-like objects in newer versions; wrapping to be safe
    try:
        return joblib.load(file_obj)
    except Exception:
        try:
            # If file_obj is a path string, open it
            if isinstance(file_obj, (str,)):
                with open(file_obj, "rb") as f:
                    return pickle.load(f)
            # else assume file-like
            file_obj.seek(0)
            return pickle.load(file_obj)
        except Exception as e:
            raise


model = None
if uploaded_file is not None:
    try:
        model = load_model_from_file(uploaded_file)
        st.sidebar.success("Model loaded from upload")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

elif model_path:
    try:
        model = load_model_from_file(model_path)
        st.sidebar.success(f"Model loaded from {model_path}")
    except Exception as e:
        st.sidebar.warning(f"Could not load model from path: {e}")

if model is not None:
    st.sidebar.write("**Model type:**")
    try:
        st.sidebar.write(type(model))
    except Exception:
        pass


st.write("Enter wine physicochemical features below and click Predict.")

# Typical features for the UCI Wine Quality dataset in expected order
# defaults = { # bad
#     "fixed acidity": 7.4,
#     "volatile acidity": 0.70,
#     "citric acid": 0.00,
#     "residual sugar": 1.9,
#     "chlorides": 0.076,
#     "free sulfur dioxide": 11.0,
#     "total sulfur dioxide": 34.0,
#     "density": 0.9978,
#     "pH": 3.51,
#     "sulphates": 0.56,
#     "alcohol": 9.4,
# }

defaults = { # 8
    "fixed acidity": 7.9,
    "volatile acidity": 0.350,
    "citric acid": 0.460,
    "residual sugar": 3.600,
    "chlorides": 0.0780,
    "free sulfur dioxide": 15.0,
    "total sulfur dioxide": 37.0,
    "density": 0.997300,
    "pH": 33.350,
    "sulphates": 0.860,
    "alcohol": 12.80,
}

with st.form("input_form"):
    fa = st.number_input("Fixed acidity (g/dm³)", value=defaults["fixed acidity"], format="%.3f")
    va = st.number_input("Volatile acidity (g/dm³)", value=defaults["volatile acidity"], format="%.3f")
    ca = st.number_input("Citric acid (g/dm³)", value=defaults["citric acid"], format="%.3f")
    rs = st.number_input("Residual sugar (g/dm³)", value=defaults["residual sugar"], format="%.3f")
    cl = st.number_input("Chlorides (g/dm³)", value=defaults["chlorides"], format="%.4f")
    fsd = st.number_input("Free sulfur dioxide (mg/dm³)", value=defaults["free sulfur dioxide"], format="%.1f")
    tsd = st.number_input("Total sulfur dioxide (mg/dm³)", value=defaults["total sulfur dioxide"], format="%.1f")
    den = st.number_input("Density (g/cm³)", value=defaults["density"], format="%.6f")
    ph = st.number_input("pH", value=defaults["pH"], format="%.3f")
    sul = st.number_input("Sulphates (g/dm³)", value=defaults["sulphates"], format="%.3f")
    alc = st.number_input("Alcohol (% vol)", value=defaults["alcohol"], format="%.2f")
    submit = st.form_submit_button("Predict")

if submit:
    if model is None:
        st.error("No model loaded. Upload a model in the sidebar or provide a valid model path.")
    else:
        features = np.array([fa, va, ca, rs, cl, fsd, tsd, den, ph, sul, alc]).reshape(1, -1)
        # Try several prediction approaches to be robust to whether the model expects arrays or DataFrames
        prediction = None
        try:
            prediction = model.predict(features)
        except Exception as e1:
            try:
                df = pd.DataFrame(features, columns=list(defaults.keys()))
                prediction = model.predict(df)
            except Exception as e2:
                st.error(f"Model prediction failed: {e1} / {e2}")

        if prediction is not None:
            # Map 0/1 to Bad/Good labels
            label_map = {0: "Bad (Quality 3-6)", 1: "Good (Quality 7-8)"}
            
            # If classifier with probabilities
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features)
                    pred_val = int(prediction[0])
                    label = label_map.get(pred_val, pred_val)
                    
                    st.success(f"Predicted class: {label}")
                    st.write("Probabilities:")
                    # Create a nice dataframe for probabilities
                    proba_df = pd.DataFrame(proba, columns=["Bad", "Good"])
                    st.dataframe(proba_df)
                except Exception:
                    st.success(f"Predicted class: {prediction[0]}")
            else:
                try:
                    val = int(prediction[0])
                    label = label_map.get(val, val)
                    st.success(f"Predicted quality: {label}")
                except Exception:
                    st.success(f"Prediction: {prediction[0]}")

st.markdown("---")
st.markdown("If your model expects a different feature order or has a preprocessing pipeline, make sure the pipeline is saved together with the final estimator (e.g., via `sklearn.pipeline.Pipeline`).")
