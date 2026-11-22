# Land Use & Land Cover Classification with Sentinel-2 (EuroSAT)
**F21DL Data Mining and Machine Learning – Group Coursework 2026/27**

**Research question:** How does the data representation (RGB images vs 13-band spectral features)
affect classification performance across the 10 EuroSAT land-cover classes?

| Hypothesis | How we test it | Where the answer is |
|---|---|---|
| **H1** Spectral advantage: extra bands help classes with similar colours | Same models on *all 13 bands* vs *visible bands only*; RGB CNN vs MS CNN | `h1_all_vs_visible.png`, `compare_per_class_f1.png` |
| **H2** Representation matters: image models keep spatial detail the CSV loses | CNN (images) vs tabular models (summary statistics) on the same test patches + McNemar test | `compare_macro_f1.png`, `mcnemar_tests.csv` |
| **H3** A VAE can reconstruct / sample plausible patches | Convolutional VAE on RGB training patches | `vae_*.png`, `vae_summary.json` |

---

## 1. Setup (do once)

Python 3.11 or newer.

```bash
git clone https://github.com/ankittoshi89-max/F21DL---Data-Mining-and-Machine-Learning-Group-Project.git
cd F21DL---Data-Mining-and-Machine-Learning-Group-Project
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
No GPU? Everything runs on CPU, the CNNs just take longer (about 1–2 h total).
Faster option: upload the folder to Google Colab, set *Runtime → GPU*, and run the same commands with `!python ...`.

## 2. Run order

Run from the project root. Each script can be run alone once the earlier ones have produced their files.

| # | Command | What it does | Requirement | Main outputs |
|---|---|---|---|---|
| 1 | `python src/s01_download_data.py` | downloads + unzips EuroSAT RGB (~90 MB) and MS (~2 GB) from Zenodo | data | `data/raw/` |
| 2 | `python src/s02_prepare_index.py` | pairs RGB↔MS files, checks unreadable / wrong-shape / duplicate files, stratified 70/15/15 split shared by every model | R2 cleaning | `data/index.csv` |
| 3 | `python src/s03_build_features.py` | turns every 13-band patch into one CSV row: band mean & std + NDVI, NDWI, NDBI (32 features) | R2, tabular dataset | `data/eurosat_ms_features.csv` |
| 4 | `python src/s04_eda.py` | class balance, sample grid, pixel histograms, spectral signatures, index boxplots, correlation, PCA/t-SNE, mutual-information feature ranking | R2 EDA + feature selection | `outputs/figures/eda_*.png` |
| 5 | `python src/s05_clustering.py` | K-means + Gaussian Mixture (EM), elbow/silhouette/BIC, ARI/NMI vs true classes | R2 clustering, generative (GMM) | `cluster_*.png`, `clustering.json` |
| 6 | `python src/s06_baselines.py` | Decision Tree, Gaussian NB, Logistic Regression, k-NN, Perceptron, Random Forest – tuned on validation, on "all" and "visible" features; tree rules + feature importance | R3 | `baselines_summary.csv`, `decision_tree_top.png` |
| 7 | `python src/s07_mlp_tabular.py` | MLP grid search (layers, L2, learning rate), learning curves | R4 | `mlp_*` |
| 8 | `python src/s08_cnn.py --bands rgb` | our CNN on RGB images, augmentation, early stopping, training curves, misclassified examples | R4 | `curves_cnn_*.png`, `mistakes_*.png` |
| 8b | `python src/s08_cnn.py --bands ms` | same CNN on all 13 bands (add `--cache-ms` if you have 4 GB+ free RAM) | R4 + H1 | |
| 8c | `python src/s08_cnn.py --bands rgb --arch resnet18 --epochs 15 --lr 3e-4` | fine-tuned pretrained ResNet-18 (transfer learning) | R4 fine-tuning | |
| 9 | `python src/s09_vae.py` | Variational Autoencoder: reconstructions, new samples, interpolation, latent space | generative task, H3 | `vae_*.png` |
| 10 | `python src/s10_compare.py` | one table of all models, per-class F1 heatmap, H1 chart, McNemar significance tests | Results section | `results_table.csv/.md` |
| ✓ | `pytest -q tests` | unit tests (feature maths, split has no overlap) | code quality | |

Or simply `bash run_all.sh` (Mac/Linux) / `run_all.bat` (Windows).

**Extra fine-tuning runs** (good for the report): try `--lr 3e-4 --tag lr3e-4`, `--batch-size 64 --tag bs64`,
`--weight-decay 1e-3 --tag wd1e-3`. Each run is saved separately and appears in the comparison table.

**Test the whole pipeline in 2 minutes without downloading** (fake data):
`python tests/make_fake_data.py` → run steps 2–10 → then delete `data/raw/`, `data/*.csv` and `outputs/` before the real run.

## 3. Using the models on new / live data

```bash
python src/predict.py my_patch.jpg                 # one 64x64 patch -> top-3 classes
python src/predict.py new_images/                  # a whole folder -> outputs/live_predictions.csv
python src/predict.py big_scene.png --gsd 10       # large image -> tiled land-cover map
python src/predict.py scene_13band.tif             # 13-band GeoTIFF -> MS CNN + random forest + ensemble
streamlit run app/streamlit_app.py                 # drag-and-drop demo for the viva
```
Getting fresh imagery: Copernicus Browser → Sentinel-2 **L1C** → export "True colour" at **10 m** resolution
(for the RGB model) or all 13 bands as GeoTIFF in EuroSAT band order (B01…B12, B8A last) for the MS models.
If the image is at another resolution, give it with `--gsd` so it is resampled to 10 m/pixel.
Low-confidence predictions are flagged – new images can differ from EuroSAT (season, sensor processing, colour stretch).

## 4. Project structure
```
src/config.py            paths, seed, class names, band order
src/utils.py             loading, feature extraction, metrics + confusion matrix
src/torch_common.py      PyTorch dataset, augmentation, CNN / ResNet / VAE definitions
src/s01 ... s10          the pipeline (run in order)
src/predict.py           predictions on new images / large scenes
app/streamlit_app.py     demo app
tests/                   unit tests + fake-data generator
outputs/figures          every plot used in the report
outputs/results          metrics (json/csv), tuning tables, statistical tests
outputs/models           saved models
```

## 5. Methodology notes (why things are done this way)
- **One split for everything.** `index.csv` fixes the train/val/test patches, so RGB and MS models are compared on identical samples.
- **No leakage.** Scalers live inside sklearn Pipelines; CNN normalisation uses training statistics only; hyper-parameters are chosen on validation; the test set is used once.
- **Macro-F1** is the main metric (every class counts equally; classes are not perfectly balanced: 2,000–3,000 patches each). Accuracy, macro precision/recall and per-class F1 are also reported.
- **Augmentation**: flips and 90° rotations only – satellite tiles have no "up", and colour jitter would change spectral meaning.
- **Generative vs discriminative**: Gaussian Naive Bayes, GMM (EM) and the VAE are generative; Decision Tree, Logistic Regression, Perceptron, k-NN, Random Forest, MLP and CNN are discriminative.

## 6. Data, licence and ethics
EuroSAT (Helber et al., 2019) – Zenodo record 7711810, MIT licence; contains modified Copernicus Sentinel data.
No personal data. Generated VAE images are *not* used as labelled ground truth.

> Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS 12(7), 2217–2226.


## 7. Contributions
| Member | Role | Files |
|---|---|---|
| Faisal Shaikh | Data & EDA lead | s01, s02, s04 |
| Faraz Ahmed | Spectral + tabular ML lead | s03, s06, s05 |
| Gayathri Dinesh | Image / CNN lead | s08, torch_common |
| Anshpreet Babra | MLP + generative lead | s07, s09 |
| Ankit Toshi | Evaluation & integration lead | s10, predict.py, app, tests |
