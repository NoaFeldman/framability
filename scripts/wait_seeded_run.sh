#!/bin/bash
# ============================================================
#  Source me:  . scripts/wait_seeded_run.sh;  wait_seeded_run <model> [poll_s]
#
#  Blocks until the seeded-rate pipeline of <model> (scripts/submit_rate_seeded.sh,
#  job names rs_<model>_<stage>) has nothing queued or running.  That driver
#  keeps at most one stage queued and submits the next one seconds after the
#  previous ends, so "idle" must hold on two checks 5 minutes apart.  Returns
#  at once (after the 5-minute confirmation) if the model has no such run.
#  The trailing _ in the pattern keeps dicke_dec_N10 apart from dicke_dec_N100
#  and from dicke_dec_k025_N10.
# ============================================================
wait_seeded_run() {
    local m="$1" poll="${2:-600}" idle=0 waited=0
    while :; do
        if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -Eq "^rs_${m}_"; then
            [ "$waited" = "1" ] || echo "[wait ${m}] seeded run still active ($(date)); polling every ${poll}s"
            waited=1; idle=0
            sleep "$poll"
        else
            idle=$(( idle + 1 ))
            [ "$idle" -ge 2 ] && break
            sleep 300
        fi
    done
    echo "[wait ${m}] no rs_${m}_* jobs queued ($(date))"
}
