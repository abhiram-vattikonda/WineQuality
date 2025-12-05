# Wine Quality Streamlit App

This small Streamlit app loads a serialized model (pickle/joblib) and predicts wine quality from physicochemical features.

How to run

1. Create a virtual environment and activate it.

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run streamlit_app.py
```

Usage notes

- The sidebar lets you upload a model file or point to a local model path. The app tries `joblib` first, then `pickle`.
- For best results save a full pipeline (preprocessing + estimator) so feature scaling and column ordering are handled.
- Default feature order in the UI follows the UCI Wine Quality dataset: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.

## Jupyter Notebook

To run the `Wine_Quality.ipynb` notebook, install ipykernel:

```powershell
pip install ipykernel
```
