from src.ingestion.generate_data import generate_datasets
from src.layer1_detector.train import train_model
from src.layer1_detector.evaluator import evaluate_layer1
from src.utils.logger import logger

def main():
    logger.info("Step 1: Generating Synthetic Dataset & Held-Out Batches...")
    generate_datasets(num_samples=1200)

    logger.info("Step 2: Training Layer 1 ML Anomaly Classifier...")
    train_model()

    logger.info("Step 3: Evaluating Layer 1 ML Model Performance...")
    metrics = evaluate_layer1()
    print("\n" + "="*50)
    print("LAYER 1 ML MODEL EVALUATION RESULTS:")
    print(f"Precision : {metrics.get('ml_model', {}).get('precision', 0)*100:.2f}%")
    print(f"Recall    : {metrics.get('ml_model', {}).get('recall', 0)*100:.2f}%")
    print(f"F1-Score  : {metrics.get('ml_model', {}).get('f1_score', 0)*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
