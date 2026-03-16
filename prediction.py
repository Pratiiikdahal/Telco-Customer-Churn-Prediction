import joblib
import pandas as pd

model = joblib.load("output_models/logreg_model.sav")
transformer = joblib.load("output_models/transformer.sav")
label_encoder = joblib.load("output_models/label_encoder.sav")

def predict(input_data):
    expected_cols = transformer.feature_names_in_ # to get the order of the columns in which i used the column transformner , doing this because i will have to matchthe corder of the columns in order to get the right prediction here.
    input_df = pd.DataFrame([input_data])[expected_cols]
    transformed_data = transformer.transform(input_df)
    prediction = model.predict(transformed_data)
    prediction = label_encoder.inverse_transform(prediction)
    return prediction[0]