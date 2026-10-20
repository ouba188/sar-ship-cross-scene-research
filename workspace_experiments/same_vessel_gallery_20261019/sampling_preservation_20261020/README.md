# Native Sampling Preservation Probe

This directory extends the frozen `native_axis_probe_20261020` classifier without changing its fitted predictions. See `PROTOCOL_ZH.md` for the prospective protocol and `FINDINGS_ZH.md` for completed results when available.

Run from this directory with the existing ClearSAR environment:

```powershell
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -m unittest test_sampling.py
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -u run.py extract
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -u run.py train
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' analyze.py
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' source_objective_diagnosis.py
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -u run.py train_mse
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' analyze.py mse
```

The local dataset, frozen R50 weights, native-axis descriptors and fitted heads are prerequisites. They are intentionally not included in the research archive. `run.py` imports existing adjacent experiment helpers; it is not a standalone benchmark package.

`*.npy`, `*.npz` and `*.pt` are local caches/weights and must not be uploaded. The archive contains code, protocol, aggregate/paired tables and selected diagnostic panels only. Large observational tables may be compressed losslessly.

The trained component is a residual descriptor reader, not a newly trained spatial convolution. No test labels select arms, steps, geometry thresholds or support sets. Case selection after evaluation is explanatory, not a new training rule.
