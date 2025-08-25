import os
import pandas as pd

def save_results(model_name, results):
  data = {
      'Model': model_name,
      'Loss': results.get('Loss', None),         # Loss
      'Accuracy': results.get('Accuracy', None),     # Accuracy
      'Precision': results.get('Precision', None),    # Precision
      'Recall': results.get('Recall', None),       # Recall
      'AUC': results.get('AUC', None),
      'Threshold': results.get('Threshold', None),
  }

  df = pd.DataFrame([data])
  df = df.fillna("null")

  # Nome do arquivo CSV
  csv_file = 'results/evaluation.csv'

  if os.path.exists(csv_file):
      return df.to_csv(csv_file, mode='a', header=False, index=False)
  df.to_csv(csv_file, mode='w', index=False)