# Run test

```
python3     Modelnet40_object_classification.py  test -a
```

# Only 15 classes in ScanoobjectNN, they are 
[b'bag'=0 b'bed'=1 b'bin'=2 b'box'=3 b'cabinet'=4 b'chair'=5 b'desk'=6 b'display'=7 b'door'=8 b'pillow'=9 b'shelf'=10 b'sink'=11 b'sofa'=12 b'table'=13 b'toilet'=14] 

# Training
## In csc
```
sbatch run.sh
tensorboard --logdir=path/to/your/log-directory #view plots in tensorboard
squeue -u $USER_NAME #view jobs in csc
``` 