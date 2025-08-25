import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def find_best_threshold(model, validation_data, validation_labels, metric='f1'):
    # Obter as probabilidades contínuas das previsões
    probabilities = model.predict(validation_data)
    thresholds = np.arange(0.1, 1.0, 0.1)  # Intervalo de thresholds para testar
    best_threshold = 0.5
    best_metric_value = 0

    # Escolher a métrica de acordo com o parâmetro
    metric_function = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    }.get(metric, f1_score)  # Padrão para F1 Score

    # Testar cada threshold
    for threshold in thresholds:
        # Converter probabilidades em rótulos binários com o threshold atual
        predictions = (probabilities > threshold).astype("int32")
        
        # Calcular a métrica para o threshold atual
        metric_value = metric_function(validation_labels, predictions)
        
        # Verificar se a métrica é a melhor até agora
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    print(f"Melhor threshold: {best_threshold} com {metric}: {best_metric_value}")
    return best_threshold
