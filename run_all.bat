@echo off
REM Windows version of run_all.sh
python src\s01_download_data.py || exit /b
python src\s02_prepare_index.py || exit /b
python src\s03_build_features.py || exit /b
python src\s04_eda.py || exit /b
python src\s05_clustering.py || exit /b
python src\s06_baselines.py || exit /b
python src\s07_mlp_tabular.py || exit /b
python src\s08_cnn.py --bands rgb || exit /b
python src\s08_cnn.py --bands ms || exit /b
python src\s08_cnn.py --bands rgb --arch resnet18 --epochs 15 --lr 3e-4 || exit /b
python src\s09_vae.py || exit /b
python src\s10_compare.py || exit /b
pytest -q tests
echo All done - figures in outputs\figures, numbers in outputs\results
