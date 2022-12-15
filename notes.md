# Sherlock

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

Run `migrate.sh` to transfer all the needed files to Sherlock. Once logged
into Sherlock rn
```
srun submit.sh
```

# Multiprocessing/Threading

* [ThreadPool vs Pool in Python multiprocessing](https://stackoverflow.com/questions/46045956/whats-the-difference-between-threadpool-vs-pool-in-the-multiprocessing-module)
* [CPU-bound vs I/O-bound](https://stackoverflow.com/questions/868568/what-do-the-terms-cpu-bound-and-i-o-bound-mean#:~:text=CPU%20bound%20means%20the%20program,the%20bottleneck%20and%20eliminate%20it.)
* [Python multiprocessing with Slurm](https://stackoverflow.com/questions/39974874/using-pythons-multiprocessing-on-slurm)
* [What are nodes in a cluster?](https://www.ibm.com/docs/en/was-nd/8.5.5?topic=servers-introduction-clusters)
* [#SBATCH directives syntax](https://slurm.schedmd.com/sbatch.html)
* [sbatch vs srun](https://stackoverflow.com/questions/43767866/slurm-srun-vs-sbatch-and-their-parameters)
* [nodes vs tasks vs cpus vs cores](https://login.scg.stanford.edu/faqs/cores/)