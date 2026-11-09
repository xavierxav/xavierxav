# Xavier Desprez

**Applied ML — computer vision and federated learning.**

Engineer from Mines Paris – PSL (Data/AI option), currently MBA at the Collège des
Ingénieurs. I build and evaluate ML systems where the data is distributed, scarce,
or sensitive — and I measure them against a baseline.

## Selected work

**[Simple-VFL](https://github.com/xavierxav/Simple-VFL)** — one-shot vertical federated
learning with frozen DINOv2 features. Each party runs its own encoder and uploads one
1536-d vector; the coordinator never touches the pixels.
`82.47 ± 0.70%` accuracy / `95.08 ± 0.17` AUC on VertiBench's official test set, against
the published SplitNN baseline at `81.17 ± 0.35%` — with 3 RGB bands instead of 13.

**Owkin data challenge** — cancer detection on 344 histology slides from three hospitals
(305 patients, 1,000 tiles per slide). Benchmarked pooling and model choices under
patient-level and leave-one-hospital-out validation: `0.702` public ROC AUC against a
`0.631` mean-pool baseline.

**ArcelorMittal data challenge — 1st place** — 16,557 chimney images labelled by emission
level. Label-efficient pipeline: day/night normalization, color distance to the mean
plume, three-frame temporal windows, and autoencoder-based anomaly detection.

**[Cantateo](https://github.com/xavierxav/cantateo)** — Django application for liturgical
psalms with synchronized Web Audio tracks, deployed on a Linux VPS with PostgreSQL,
Redis and Celery. Live at [cantateo.fr](https://www.cantateo.fr).

## Toolbox

`PyTorch` · `DINOv2` · `scikit-learn` · `Python` · `Docker` · `PostgreSQL` · `Linux`

Contact: [xav.desprez@gmail.com](mailto:xav.desprez@gmail.com) · [LinkedIn](https://www.linkedin.com/in/xavierdesprez/)
