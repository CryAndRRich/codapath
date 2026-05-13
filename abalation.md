4 ablation studies:
- CODAPath but backbone only has PLIP
- CODAPath but backbone only has BioMedCLIP,
- CODAPath but finetune omits LoRa and only updates the projection layer with CLS,
- CODAPath but omits contrastive loss.

Sampling ablations:
- `codapath_no_uncertainty`: keeps CODAPath spatial coverage and removes the uncertainty term.
- `codapath_no_spatial_coverage`: keeps CODAPath uncertainty ranking and removes the spatial coverage term.

