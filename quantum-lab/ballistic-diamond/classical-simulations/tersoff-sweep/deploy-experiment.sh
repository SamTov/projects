# Run a large sweep for tin implantation in diamond.

annealing_temperatures=(300)  # (300 700 1000 1400 1600)
strike_angles=(0 3 7)  # (0 15 30 45 60 75)
strike_velocities=(1803.08828307 2549.95190407 3606.17656613 3123.0405168 4031.82797037 5256.86052036)
ensembles=100

for velocity in "${strike_velocities[@]}"; do
  mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}
  mkdir velocity-${velocity}
  for temperature in "${annealing_temperatures[@]}"; do
    mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}/temperature-${temperature}
    mkdir velocity-${velocity}/temperature-${temperature}
    for i in $(seq 0 $((${ensembles}-1))); do
        for angle in "${strike_angles[@]}"; do
            mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}/temperature-${temperature}/angle-${angle}-${i}
            mkdir velocity-${velocity}/temperature-${temperature}/angle-${angle}-${i}
            cp simulate.lmp submit.sh velocity-${velocity}/temperature-${temperature}/angle-${angle}-${i}
            cd velocity-${velocity}/temperature-${temperature}/angle-${angle}-${i}
            sed -i "s/ANNEALTEMP/${temperature}/g" simulate.lmp
            sed -i "s/STRIKEANGLE/${angle}/g" simulate.lmp
            sed -i "s/STRIKESPEED/${velocity}/g" simulate.lmp
            sed -i "s/ENSEMBLE/${i}/g" simulate.lmp

            sbatch submit.sh
            cd ../../..
        done
    done
  done
done