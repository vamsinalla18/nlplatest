from inference.disease_predictor import DiseasePredictor


predictor = DiseasePredictor()

symptoms = "fever headache rash"

predictions = predictor.predict(symptoms)

for disease, score in predictions:

    print("\nPredicted Disease:", disease)
    print("Score:", score)