#!/usr/bin/env bash
# Runs the whole project in order. Takes a while on CPU (the CNNs are the slow part).
set -e
python src/s01_download_data.py
python src/s02_prepare_index.py
python src/s03_build_features.py
python src/s04_eda.py
python src/s05_clustering.py
python src/s06_baselines.py
python src/s07_mlp_tabular.py
python src/s08_cnn.py --bands rgb
python src/s08_cnn.py --bands ms
python src/s08_cnn.py --bands rgb --arch resnet18 --epochs 15 --lr 3e-4
python src/s09_vae.py
python src/s10_compare.py
pytest -q tests
echo "All done - figures in outputs/figures, numbers in outputs/results"
