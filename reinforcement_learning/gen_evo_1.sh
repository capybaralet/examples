filename="job_evo_1.sh"
cp template.sh $filename

for SEED in {1..10}; do
    for AGENTS in 1; do
        for LR in 1e-3 5e-3 1e-2 5e-2; do
            echo "srun --label python evo_reinforce.py --savedir /data/lisa/exp/zac/results/evo/ps${PSURV}a${AGENTS}seed${SEED}lr${LR} -ps ${PSURV} -a ${AGENTS} --seed ${SEED} --lr ${LR}" >> $filename
        done
    done
done

for SEED in {1..10}; do
    for PSURV in 0.5 0.75; do
        for AGENTS in 10 20 30; do
            for LR in 1e-3 5e-3 1e-2 5e-2; do
                echo "srun --label python evo_reinforce.py --savedir /data/lisa/exp/zac/results/evo/ps${PSURV}a${AGENTS}seed${SEED}lr${LR} -ps ${PSURV} -a ${AGENTS} --seed ${SEED} --lr ${LR}" >> $filename
            done
        done
    done
done

echo "source deactivate" >> $filename
