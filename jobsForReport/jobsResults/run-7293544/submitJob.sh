#PBS -l nodes=1:ppn=12
#PBS -l walltime=100:00:00
#PBS -l mem=164GB
#PBS -N kmeans
#PBS -M mc3784@nyu.edu
#PBS -j oe
#PBS -m e


module purge

SRCDIR=$HOME/grasp-lift-eeg-detection/jobsForReport
RUNDIR=$SCRATCH/jobsForReport/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load virtualenv/12.1.1;
module load scipy/intel/0.16.0

source /home/mc3784/kmeans/venv2/bin/activate
python onlineKmeansComparison.py
