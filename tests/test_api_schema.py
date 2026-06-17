from mlproject.api import Features, PredictionOut


def test_features_schema_accepts_valid_payload():
    payload = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    features = Features(**payload)

    assert features.tenure == 12
    assert features.InternetService == "Fiber optic"


def test_prediction_output_schema():
    output = PredictionOut(prediction=1, probability=0.75)

    assert output.prediction == 1
    assert output.probability == 0.75