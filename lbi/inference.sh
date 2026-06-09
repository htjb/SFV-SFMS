#!/usr/bin/env bash
# Launch parallel inference jobs (blackjax-inference.py).
# Requires trained emulators to exist for each combination.
#
# Usage: bash lbi/inference.sh
# Monitor: tail -f lbi/inference-logs/<name>.log

PYTHON="python"
LOGDIR="lbi/inference-logs"
mkdir -p "$LOGDIR"

KERNELS=("ext-reg" "sho")
ZVALS=("3.5" "4.5" "5.5" "6.5" "7.5")
MASS_BINS=("8_8.5" "8.5_9" "9_9.5" "9.5_10")

BATCH_SIZE=8
PIDS=()
FOREGROUND=${FOREGROUND:-0}

for KERNEL in "${KERNELS[@]}"; do
    for ZVAL in "${ZVALS[@]}"; do
        for MB in "${MASS_BINS[@]}"; do
            NAME="${KERNEL}_z${ZVAL}_${MB}"
            LOG="$LOGDIR/${NAME}.log"
            echo "Launching $NAME -> $LOG"

            if [[ "$FOREGROUND" == "1" ]]; then
                $PYTHON lbi/blackjax-inference.py $KERNEL $MB $ZVAL 2>&1 | tee "$LOG"
            else
                $PYTHON lbi/blackjax-inference.py $KERNEL $MB $ZVAL > "$LOG" 2>&1 &
                PIDS+=($!)

                if (( ${#PIDS[@]} >= BATCH_SIZE )); then
                    echo "Waiting for batch of $BATCH_SIZE jobs..."
                    wait "${PIDS[@]}"
                    PIDS=()
                    echo "Batch done."
                fi
            fi
        done
    done
done

if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for final ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
fi

echo "All inference jobs complete."
