from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Função perform_grid_search (caso ainda não esteja definida)
def perform_grid_search(pipeline, param_grid, X, y, scoring='recall', cv=5, verbose=1):
    """
    Realiza o GridSearchCV para otimizar os hiperparâmetros de um pipeline.

    Args:
        pipeline (Pipeline): Pipeline com pré-processador e modelo.
        param_grid (dict): Dicionário com os hiperparâmetros a serem testados.
        X (DataFrame): Conjunto de dados (features).
        y (Series): Conjunto de dados (target).
        scoring (str): Métrica de avaliação (padrão: 'recall').
        cv (int): Número de dobras para validação cruzada (padrão: 5).
        verbose (int): Nível de detalhamento do progresso (padrão: 1).

    Returns:
        dict: Melhores parâmetros encontrados.
        float: Melhor pontuação obtida.
        Pipeline: Pipeline treinado com os melhores parâmetros.
    """
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    print("Best params:", grid_search.best_params_)
    print(f"Best {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_

def plot_results(y_true, y_pred, y_proba):


	# Classification Report
	print("Classification Report:")
	print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))

	# Confusion Matrix
	cm = confusion_matrix(y_true, y_pred)
	auc = roc_auc_score(y_true, y_proba)
	fpr, tpr, thresholds = roc_curve(y_true, y_proba)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	sns.heatmap(cm, annot=True, fmt='d',
				xticklabels=['No Churn', 'Churn'],
				yticklabels=['No Churn', 'Churn'],
				cmap='Blues', ax=axes[0])
	axes[0].set_xlabel('Predicted')
	axes[0].set_ylabel('Real')
	axes[0].set_title('Confusion Matrix')

	# ROC Curve
	axes[1].set_ylim([0.0, 1])
	axes[1].set_xlim([0.0, 1])
	axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', color='#003F7E')
	axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Random')
	axes[1].set_xlabel('False Positive Rate (FPR)')
	axes[1].set_ylabel('True Positive Rate (TPR)')
	axes[1].set_title('ROC Curve')
	axes[1].legend(loc='lower right')
	plt.tight_layout()
	plt.show()