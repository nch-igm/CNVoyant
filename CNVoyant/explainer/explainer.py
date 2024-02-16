import pandas as pd
# import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class Explainer:

    def __init__(self, chrom, start, end, var_type, output_dir, classifier):

        # Get root path
        self.chrom = chrom
        self.start = start
        self.end = end
        self.var_type = var_type
        self.output_dir = output_dir
        self.classifier = classifier


    def explain(self):

        # Predict the probabilities
        probabilities = model.predict_proba([data_instance])[0]

        # Donut plot
        plt.figure(figsize=(6, 6))
        plt.pie(probabilities, labels=model.classes_, wedgeprops=dict(width=0.3), startangle=140, autopct='%1.1f%%')
        plt.title('Prediction Probability')

        # SHAP values
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(data_instance)

        # # Force plot for a specific class (e.g., the predicted class)
        # predicted_class = model.predict([data_instance])[0]
        # class_index = list(model.classes_).index(predicted_class)
        # shap.force_plot(explainer.expected_value[class_index], shap_values[class_index][0], data_instance, matplotlib=True)
