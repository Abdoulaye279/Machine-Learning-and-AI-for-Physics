# Machine learning on particle detector data

Regression and classification on four detector datasets: energy calibration of a scintillator, beta / cosmic separation, pulse / noise discrimination and alpha / proton identification. Each model is compared with a simple physics-based baseline, and the irreducible errors are explained from the physics.

## Key results

| Task | Best model | Score | Baseline |
|---|---|---|---|
| Energy calibration (regression) | Linear, ADC channel + temperature | RMSE 5.6 keV, R² 0.99987 | ADC only: RMSE 6.6 keV |
| Beta vs cosmic | Neural network / random forest | Accuracy 0.929, F1 0.934 | Energy and angle cut: 0.909 |
| Pulse vs noise | Logistic regression, amplitude + width | Accuracy 0.954 | Amplitude only: 0.868 |
| Alpha vs proton | Logistic regression on log E, log depth | Accuracy 0.987 | Decision tree 0.983, linear SVM 0.970 |

- Beta / cosmic cannot reach 100%: about 17-21 of 146 test cosmics have the energy and direction of a beta. A third measurement (veto detector) would be needed.
- Alpha / proton: the border follows depth ∝ E^1.64 (fitted). It is straight in log space, so a linear model on the logs beats the non-linear ones.

## Data

| File | Events | Features | Target | After cleaning |
|---|---|---|---|---|
| `calibration.csv` | 203 | ADC channel, temperature | energy (keV) | 192 |
| `beta_cosmic.csv` | 1,203 | energy (MeV), cos θ | beta = 1 | 1,188 (605 beta, 583 cosmic) |
| `pulse_noise.csv` | 803 | amplitude (mV), width (ns) | pulse = 1 | 788 (335 pulse, 453 noise) |
| `alpha_proton.csv` | 1,203 | energy (MeV), depth (µm) | alpha = 1 | 1,188 (570 alpha, 618 proton) |

Cleaning: duplicated rows and missing values removed, unphysical values cut (negative ADC, energy or depth, |cos θ| > 1, temperature outside 0-50 °C).

## Methodology

- Train / test split 80/20 (calibration) or 75/25 stratified (classification), `random_state=42`.
- Classification: model selection by stratified 5-fold cross-validation on the training set only; the test set is used once at the end.
- Features standardised for kNN, SVM and neural networks.
- Linear and logistic regression also solved by hand with gradient descent, to check against scikit-learn.
- Neural networks in Keras with fixed seeds.

| Family | Models |
|---|---|
| Linear | Linear regression, logistic regression |
| Distance / kernel | kNN (k = 1-51), SVM linear and RBF |
| Trees | Decision tree (depth 1-10, unlimited), random forest |
| Neural | Dense ReLU networks, 1 to 32 hidden units, 2 layers for alpha / proton |

## Results

### Energy calibration (test set, 39 peaks)

| Model | MAE (keV) | RMSE (keV) | R² |
|---|---|---|---|
| ADC only | 4.76 | 6.57 | 0.99983 |
| ADC + temperature | 4.32 | 5.63 | 0.99987 |

E = 34.1 + 0.796 × ADC - 1.32 × T: at a fixed ADC channel the fitted energy shifts by -1.32 keV per °C. Gradient descent gives the same coefficients as least squares to 12 digits.

### Beta vs cosmic (test set, 297 events)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Cut E < 2.3 MeV, cos θ > 0.75 | 0.909 | 0.852 | 0.993 | 0.917 |
| kNN (k = 21) | 0.926 | 0.877 | 0.993 | 0.932 |
| SVM RBF | 0.926 | 0.877 | 0.993 | 0.932 |
| Tree (depth 2) | 0.919 | 0.876 | 0.980 | 0.925 |
| Random forest | 0.929 | 0.896 | 0.974 | 0.933 |
| Neural network (8 ReLU units) | 0.929 | 0.892 | 0.980 | 0.934 |

The deeper trees overfit (CV 0.915 at depth 2, 0.862 unlimited). A 2-unit network fails on one seed out of five (validation 0.475).

### Pulse vs noise (test set, 197 signals)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic, amplitude only (threshold 44.9 mV) | 0.868 | 0.863 | 0.821 |
| Logistic, amplitude + width | 0.954 | 0.963 | 0.929 |

### Alpha vs proton (test set, 297 particles)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| kNN (k = 1) | 0.980 | 0.972 | 0.986 | 0.979 |
| SVM linear | 0.970 | 0.965 | 0.972 | 0.968 |
| SVM RBF | 0.976 | 0.953 | 1.000 | 0.976 |
| Tree (depth 10) | 0.983 | 0.986 | 0.979 | 0.982 |
| Random forest | 0.976 | 0.972 | 0.979 | 0.975 |
| Neural network (2 × 16 ReLU) | 0.976 | 0.972 | 0.979 | 0.975 |
| Logistic regression on log E, log depth | 0.987 | | | |

Standardising the inputs takes kNN from 0.943 to 0.983. With 297 test particles one error is 0.003 of accuracy: the gaps between the non-linear models (2-5 particles) are not significant.

## Limitations

- Small datasets (200-1,200 events), one test split each.
- Only two input features per task; the beta / cosmic overlap is a limit of the inputs, not of the models.
- Neural network results depend on the seed for very small networks.

## Reproducing

```
pip install numpy pandas matplotlib scikit-learn tensorflow
jupyter notebook
```

| Notebook | Content | Runtime |
|---|---|---|
| `Energy_Calibration_and_BetaCosmic_Separation` | Calibration regression, beta / cosmic classification | ~3 min |
| `PulseNoise_and_AlphaProton_Classification` | Pulse / noise, alpha / proton | ~1 min |

Each notebook opens in Colab from its badge; the first cell downloads the repository and the data.
