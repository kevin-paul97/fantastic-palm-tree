#!/usr/bin/env bash
set -euo pipefail

# Cross-validation grid search over batch size and learning rate.
# Results are logged to cross_validation_results.csv.

BATCH_SIZES=(16 32 64)
LEARNING_RATES=(0.01 0.001 0.0001)
EPOCHS=20
RESULTS_FILE="cross_validation_results.csv"

echo "batch_size,learning_rate,best_val_loss,model_path" > "$RESULTS_FILE"

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    echo "============================================"
    echo "Training: batch_size=$bs  lr=$lr  epochs=$EPOCHS"
    echo "============================================"

    model_tag="bs${bs}_lr${lr}"
    log_output="outputs/cv_${model_tag}.log"
    mkdir -p outputs

    # Train and capture output
    if python3 main.py train regressor \
        --batch-size "$bs" \
        --lr "$lr" \
        --epochs "$EPOCHS" \
        --no-tensorboard 2>&1 | tee "$log_output"; then

      # Extract best validation loss from log (macOS-compatible)
      best_val=$(sed -n 's/.*Best validation loss: \([0-9.]*\).*/\1/p' "$log_output" | tail -1)
      best_val="${best_val:-N/A}"
      model_path="models/regressor_final.pth"

      echo "$bs,$lr,$best_val,$model_path" >> "$RESULTS_FILE"
      echo ">> Result: batch_size=$bs  lr=$lr  best_val_loss=$best_val"

      # Keep a copy of the best model for this run
      cp "$model_path" "models/regressor_${model_tag}.pth"
    else
      echo "$bs,$lr,FAILED," >> "$RESULTS_FILE"
      echo ">> FAILED: batch_size=$bs  lr=$lr"
    fi

    echo ""
  done
done

echo "============================================"
echo "Cross-validation complete. Results:"
echo "============================================"
column -t -s',' "$RESULTS_FILE"

# Print the best run
echo ""
echo "Best run:"
tail -n +2 "$RESULTS_FILE" | grep -v FAILED | sort -t',' -k3 -n | head -1
