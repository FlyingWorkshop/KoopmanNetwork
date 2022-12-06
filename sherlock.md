Connect to Sherlock with: 
```
ssh loganmb@login.sherlock.stanford.edu
```
and then use your SUID password. 

To copy files from local directory to Sherlock:
```
scp <filename> loganmb@login.sherlock.stanford.edu:
```

To run a job
```
sbatch submit.sh
```

Check queue status:
```
squeue -u $USER
```
Interpret task status: https://curc.readthedocs.io/en/latest/running-jobs/squeue-status-codes.html

