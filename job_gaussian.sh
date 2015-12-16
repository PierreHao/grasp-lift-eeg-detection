#!/bin/bash
#PBS -l nodes=1:ppn=12
#PBS -l walltime=10:00:00
#PBS -l mem=164GB
#PBS -N gaussian_vlad
#PBS -M ajr619@nyu.edu
#PBS -j oe

module purge

SRCDIR=$HOME/workspace/grasp-lift-eeg-detection/
RUNDIR=$SCRATCH/grasp-lift-eeg-detection/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR

#cp -R $SRCDIR/* $RUNDIR
cp -R $SRCDIR/data $RUNDIR
cp $SRCDIR/vlad.py $RUNDIR
cp $SRCDIR/linear_vlad_pipeline.py $RUNDIR
cp $SRCDIR/gaussian_vlad_pipeline.py $RUNDIR
cp $SRCDIR/requirements.txt $RUNDIR

module load virtualenv/12.1.1;
module load scipy/intel/0.16.0
module load psutil/intel/2.1.3

virtualenv .venv

source .venv/bin/activate;

pip install -r requirements.txt
cd $RUNDIR

python gaussian_vlad_pipeline.py 3 
