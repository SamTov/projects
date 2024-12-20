# Run a large sweep for tin implantation in diamond.

annealing_temperatures=(300 700 1000 1400 1600)
strike_angles=(0 15 30 45 60 75)
strike_velocities=(1803.08828307 2549.95190407 3606.17656613 5256.86052036)

for velocity in "${strike_velocities[@]}"; do
  mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}
  mkdir velocity-${velocity}
  for temperature in "${annealing_temperatures[@]}"; do
    mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}/temperature-${temperature}
    mkdir velocity-${velocity}/temperature-${temperature}
    for angle in "${strike_angles[@]}"; do
        mkdir /work/stovey/ballistic-diamond/tersoff-sweep/velocity-${velocity}/temperature-${temperature}/angle-${angle}
        mkdir velocity-${velocity}/temperature-${temperature}/angle-${angle}
        cp simulate.lmp submit.sh velocity-${velocity}/temperature-${temperature}/angle-${angle}
        cd velocity-${velocity}/temperature-${temperature}/angle-${angle}
        sed -i "s/ANNEALTEMP/${temperature}/g" simulate.lmp
        sed -i "s/STRIKEANGLE/${angle}/g" simulate.lmp
        sed -i "s/STRIKESPEED/${velocity}/g" simulate.lmp
        sbatch submit.sh
        cd ../../..
    done
  done
done