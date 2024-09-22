#!/bin/bash
#SBATCH -A <project>
#SBATCH -J <job-name>
#SBATCH -o "outjob_%j"
#SBATCH -N 15
#SBATCH -p batch
#SBATCH -t 10:00:00

echo "===============STARTING TIME==============="
date
    
nnodes=$SLURM_JOB_NUM_NODES
ncores=32
job_limit=$(($nnodes * ncores))

if [[$job_limit >= 640]]; then
  echo "Job limit = " $job_limit ", which is larger than allowed job limit in Andes (20*32=640). Recommend to reduce number of nodes requested and increase the time."
fi

work_dir="/gpfs/alpine/proj-shared/<path-to-repo>/sst-urans-timeseries-ml"
echo $work_dir
cd $work_dir

module load python
source activate base
conda activate /gpfs/alpine/<path-to-conda-env>/sst-urans-ml

# echo "==========running data script=========="
# jsrun -r 1 -n 1 -a 1 -c 1 -g 1 python DataAcqu_LSTM.py --machine_dir Data_raw --case 1 --seq_len 16 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --savedata --HDdir $work_dir
# 
# echo "==========running NN training script=========="
# jsrun -r 1 -n 1 -a 1 -c 1 -g 1 python nnTraining.py --casenum 1 --normEnergy --interpIO --set_dt_seq --dt_T 0.5 --seq_len 16 --nepoch 50 --batch_size 10 --saveNN --HDdir $work_dir

echo "*********prallel parametric training***********"

seq_len=(8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128)
batch_size=(10 10 10 10 10 10 10 10 10 50 50 50 50 50 50 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100)
dt_T=(0.02 0.05 0.07 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0)

len_seq=${#seq_len[@]}
len_T=${#dt_T[@]}

echo "Length of seq: " $len_seq "     Length of T: " $len_T  "     Number of jobs: " $(($len_T*$len_seq))

echo "==========================Running data generation=========================="
counter=0
for (( i=0 ; i<$len_T ; i++ ));
do
  for (( j=0 ; j<$len_seq ; j++ ));
  do
    echo "==========================seq_len = "${seq_len[$j]} "; dt_T = "${dt_T[$i]}"=========================="
    srun -N 1 -n 1 -c 1 python DataAcqu_LSTM.py --Datadir Data_raw --case 13 --target_T 1.0 --seq_len_T ${seq_len[$j]}  --normEnergy --interpIO --set_dt_seq --dt_T ${dt_T[$i]} --savedata --HDdir $work_dir &
    counter=$(($counter + 1))
    echo "counter: " $counter
    if [[ $counter == $job_limit ]]; then
      wait
      echo "*******************Waiting for ports to cleanup*********************"
      sleep 60
      echo "*****************************Continuing*****************************"
      counter=0
    fi
  done
done

wait
echo "==========================Finished data generation=========================="
echo "*******************Waiting for ports to cleanup*********************"
sleep 60

echo "==========================Running NN training=========================="
counter=0
for (( i=0 ; i<$len_T ; i++ ));
do
  for (( j=0 ; j<$len_seq ; j++ ));
  do
    echo "==========================seq_len = "${seq_len[$j]} "; dt_T = "${dt_T[$i]}"=========================="
    srun -N 1 -n 1 -c 1 python nnTraining.py --casenum 13 --target_T 1.0 --seq_len_T ${seq_len[$j]} --normEnergy --interpIO --set_dt_seq --dt_T ${dt_T[$i]} --nepoch 4000 --batch_size ${batch_size[$j]} --saveNN --HDdir $work_dir &
    counter=$(($counter + 1))
    echo "counter: " $counter
    if [[ $counter == $job_limit ]]; then
      wait
      echo "*******************Waiting for ports to cleanup*********************"
      sleep 60
      echo "*****************************Continuing*****************************"
      counter=0
    fi

  done
done

wait
echo "==========================Finished NN training=========================="


echo "*********finished running parallel jobs***********"

echo "===============ENDING TIME==============="
date



