"""Generate the synthetic dataset, train Layer 1, and report honest metrics.

Run this once before starting the API or either dashboard: the dataset, the
model and the held-out verification batches are all build artifacts and are
not committed to the repository.
"""
from __future__ import annotations

import argparse
import sys

from src.ingestion.generate_data import generate_datasets
from src.layer1_detector.evaluator import evaluate_layer1
from src.layer1_detector.train import train_model
from src.utils import config
from src.utils.logger import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1200,
                        help="number of training samples to generate")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="RNG seed, for reproducible datasets")
    parser.add_argument("--skip-generate", action="store_true",
                        help="train on the existing dataset instead of regenerating")
    args = parser.parse_args()

    if not args.skip_generate:
        logger.info("Step 1/3: generating synthetic dataset and held-out batches...")
        summary = generate_datasets(num_samples=args.samples, seed=args.seed)
        logger.info(
            "Generated %d samples and %d held-out batches (seed %d).",
            summary["samples"], summary["heldout_batches"], summary["seed"],
        )
    else:
        logger.info("Step 1/3: skipped generation (--skip-generate).")

    logger.info("Step 2/3: training the Layer 1 anomaly classifier...")
    training = train_model()
    if training is None:
        logger.error("Training failed; aborting before evaluation.")
        return 1

    logger.info("Step 3/3: evaluating Layer 1 on the held-out split...")
    metrics = evaluate_layer1()
    if not metrics:
        logger.error("Evaluation produced no metrics.")
        return 1

    model = metrics["ml_model"]
    baseline = metrics["rule_baseline"]

    print("\n" + "=" * 68)
    print("LAYER 1 EVALUATION (held-out split only)")
    print("=" * 68)
    print(f"Rows evaluated   : {metrics['evaluated_rows']}  (held_out={metrics['held_out']})")
    print(f"Train accuracy   : {training['train_accuracy'] * 100:.2f}%")
    print(f"Test accuracy    : {training['test_accuracy'] * 100:.2f}%")
    print("-" * 68)
    print(f"{'':<16}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    print(f"{'ML model':<16}{model['precision'] * 100:>11.2f}%"
          f"{model['recall'] * 100:>11.2f}%{model['f1_score'] * 100:>11.2f}%")
    print(f"{'Rule baseline':<16}{baseline['precision'] * 100:>11.2f}%"
          f"{baseline['recall'] * 100:>11.2f}%{baseline['f1_score'] * 100:>11.2f}%")
    print("=" * 68)
    print(f"\nNote: {metrics['split_note']}")
    print(f"\nCaveat: {metrics['caveat']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
