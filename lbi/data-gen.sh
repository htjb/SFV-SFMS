#!/usr/bin/env bash
# Launch parallel data-gen jobs in a single tmux session.
# Each job gets its own window so you can monitor progress.
#
# Usage: bash lbi/data-gen.sh
# Attach to session: tmux attach -t data-gen
# Switch windows:    Ctrl-b n / Ctrl-b p  (next/prev)
# Kill session when done: tmux kill-session -t data-gen

PYTHON="python"   # or full path to your venv python
LOGDIR="lbi/data-gen-logs"
mkdir -p "$LOGDIR"

# kernels, redshifts, mass bins, and corresponding seed masses
KERNELS=("ext-reg" "sho")
ZVALS=("3.5" "4.5" "5.5" "6.5" "7.5")

# Seed mass as a function of redshift and mass bin.
# Fill in values from mean_mass_vs_seed_mass.py output.
get_seed_mass() {
    local zval=$1
    local mb=$2
    case "${zval}_${mb}" in
        "3.5_8_8.5")   echo "1.074e+07" ;;
        "3.5_8.5_9")   echo "1.938e+07" ;;
        "3.5_9_9.5")   echo "3.530e+07" ;;
        "3.5_9.5_10")  echo "6.734e+07" ;;
        "4.5_8_8.5")   echo "1.718e+07" ;;
        "4.5_8.5_9")   echo "3.135e+07" ;;
        "4.5_9_9.5")   echo "5.894e+07" ;;
        "4.5_9.5_10")  echo "1.129e+08" ;;
        "5.5_8_8.5")   echo "2.730e+07" ;;
        "5.5_8.5_9")   echo "5.102e+07" ;;
        "5.5_9_9.5")   echo "9.572e+07" ;;
        "5.5_9.5_10")  echo "1.913e+08" ;;
        "6.5_8_8.5")   echo "4.308e+07" ;;
        "6.5_8.5_9")   echo "8.112e+07" ;;
        "6.5_9_9.5")   echo "1.569e+08" ;;
        "6.5_9.5_10")  echo "3.074e+08" ;;
        "7.5_8_8.5")   echo "6.686e+07" ;;
        "7.5_8.5_9")   echo "1.277e+08" ;;
        "7.5_9_9.5")   echo "2.477e+08" ;;
        "7.5_9.5_10")  echo "5.065e+08" ;;
        *) echo "MISSING_SEED_MASS_${zval}_${mb}" ;;
    esac
}

MASS_BINS=("8_8.5" "8.5_9" "9_9.5" "9.5_10")
# Bin centers: 8.25, 8.75, 9.25, 9.75


BATCH_SIZE=8
PIDS=()
FOREGROUND=${FOREGROUND:-0}  # set FOREGROUND=1 to run sequentially without backgrounding

for KERNEL in "${KERNELS[@]}"; do
    for ZVAL in "${ZVALS[@]}"; do
        for MB in "${MASS_BINS[@]}"; do
            SEED=$(get_seed_mass "$ZVAL" "$MB")

            NAME="${KERNEL}_z${ZVAL}_${MB}"
            LOG="$LOGDIR/${NAME}.log"
            echo "Launching $NAME -> $LOG"

            if [[ "$FOREGROUND" == "1" ]]; then
                # Run sequentially in foreground — full CPU priority, no batching
                $PYTHON lbi/data-gen.py $KERNEL $ZVAL $MB $SEED 2>&1 | tee "$LOG"
            else
                $PYTHON lbi/data-gen.py $KERNEL $ZVAL $MB $SEED > "$LOG" 2>&1 &
                PIDS+=($!)

                # When batch is full, wait for all to finish before launching more
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

# Wait for any remaining jobs
if (( ${#PIDS[@]} > 0 )); then
    echo "Waiting for final ${#PIDS[@]} jobs..."
    wait "${PIDS[@]}"
fi

echo "All jobs complete."
