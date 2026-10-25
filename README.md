# CNC Chatter Suppression and Tool-Path Optimization
### YUKTHI 2026 Hackathon Prototype — Industrial Milling Dynamics & AI Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed-brightgreen.svg)]()

---

> [!WARNING]
> **SIMULATION & RESEARCH PROTOTYPE DISCLAIMER**
> This application is an engineering simulation and algorithmic prototype developed for the **YUKTHI 2026 Hackathon**.
> - All vibration sensor features, FFT spectra, and training datasets are **synthetically generated** using domain-physics approximations (Altintas-Budak regenerative chatter theory and forced excitation models).
> - Calculated metrics (Material Removal Rate $MRR$, cycle time delta, tool wear factor) are **illustrative trade-off models** for algorithmic demonstration.
> - This software does **NOT connect to, nor control, any physical CNC machine**. No real-world machining performance claims are made.

---

## 📌 Executive Summary

Regenerative chatter is a self-excited vibrational instability that occurs during metal cutting when the cutting tool encounters surface wave modulations left by previous tooth passes. Left unmitigated, severe chatter causes rapid tool wear, catastrophic insert breakage, spindle bearing degradation, and unacceptable workpiece surface finish.

This prototype provides an industrial-grade diagnostic and optimization engine:
1. **Real-Time Machine & Sensor Inputs**: Ingests spindle speed ($RPM$), feed rate ($v_f$), axial depth of cut ($a_p$), vibration RMS, vibration peak, and dominant spectral frequency.
2. **Machine Learning Classifier**: Employs a Scikit-Learn **Random Forest Classifier** to classify operating states into `Stable`, `Warning`, or `High Risk` with class probability breakdowns.
3. **Domain Physics Grounding**: Computes chip load per tooth ($f_z$), cutting speed ($V_c$), tooth passing frequency ($f_t$), cutting forces ($F_c$), spindle cutting power ($P_c$), and Material Removal Rate ($MRR$).
4. **Configured Boundary Validation**: Rigorously evaluates machine and tooling limits (minimum rubbing chip load, maximum breakage chip load, spindle motor continuous power, axis feed limits, and maximum spindle speed), automatically rejecting infeasible candidates with transparent diagnostic logs.
5. **Optimization Engine**: Identifies stability lobe harmonic "sweet spot" pockets ($f_{tooth} \approx f_n / k$) and generates alternative candidate machining settings.
6. **Multi-Objective Trade-Off Ranking**: Ranks candidates using a customizable Pareto utility function balancing chatter risk reduction, productivity ($MRR$), cycle time, and tool wear.
7. **Side-by-Side Comparison & Visualizations**: Delivers Stability Lobe Diagrams (SLD), synthetic FFT frequency spectra, trade-off scatter plots, and model explainability metrics.

---

## 🏗️ Architecture & Project Structure

```
cnc-chatter-optimization/
├── .gitignore                      # Git exclusion rules (venv, caches)
├── requirements.txt                # Production dependencies
├── README.md                       # Comprehensive documentation
├── app.py                          # Streamlit application UI & reactive controller
├── core/
│   ├── __init__.py
│   ├── physics.py                  # Domain physics: MRR, chip load, forces, power, SLD boundary
│   ├── synthetic_data.py           # Reproducible synthetic dataset generator with physics coupling
│   ├── model.py                    # Random Forest pipeline, metrics, probabilities & inference
│   ├── bounds.py                   # Machine & tooling constraints, feasibility bound checks
│   └── optimizer.py                # Candidate generation, bound filtering, Pareto trade-off ranking
├── visualization/
│   ├── __init__.py
│   └── plots.py                    # Matplotlib industrial visualizers (SLD, FFT, Pareto, CM, Importances)
└── tests/
    ├── __init__.py
    ├── test_physics.py             # 7 unit tests for physics equations
    ├── test_synthetic_data.py      # 5 tests for dataset schema, reproducibility & consistency
    ├── test_model.py               # 4 tests for model fitting, inference & bounds
    ├── test_bounds.py              # 5 tests for machine constraint violations
    ├── test_optimizer.py           # 2 tests for candidate generation & ranking
    ├── test_visualization.py       # 5 tests for plot rendering
    └── test_app_integration.py     # End-to-end integration across scenarios
```

---

## ⚙️ Engineering Formulations

### 1. Chip Load & Material Removal Rate
$$\text{Chip Load: } f_z = \frac{v_f}{N \cdot \text{RPM}} \quad [\text{mm/tooth}]$$
$$\text{Material Removal Rate: } MRR = \frac{a_p \cdot a_e \cdot v_f}{1000} \quad [\text{cm}^3/\text{min}]$$

### 2. Cutting Speed & Tooth Passing Frequency
$$V_c = \frac{\pi \cdot D \cdot \text{RPM}}{1000} \quad [\text{m/min}], \quad f_t = \frac{N \cdot \text{RPM}}{60} \quad [\text{Hz}]$$

### 3. Tangential Force & Spindle Power
$$F_c = K_c \cdot a_p \cdot h_m^{1 - m_c} \quad [\text{N}], \quad P_c = \frac{F_c \cdot V_c}{60000} \quad [\text{kW}]$$
where $h_m \approx f_z \cdot \sqrt{a_e / D}$ is the mean uncut chip thickness.

### 4. Regenerative Stability Boundary & Sweet Spots
Sweet spot speeds occur where tooth passing frequency harmonics align with structural resonant nodes:
$$\text{RPM}_k \approx \frac{60 \cdot f_n}{N \cdot k} \quad (k = 1, 2, 3, \dots)$$

---




## 💻 Dashboard Walkthrough

1. **Preset Scenarios**: Use the sidebar buttons to immediately load:
   - 🔴 **Chatter**: Severe regenerative vibration ($a_p = 4.2$ mm, RMS $= 3.25$ g, locked at $850$ Hz).
   - 🟡 **Warning**: Marginal stability boundary ($a_p = 2.8$ mm, RMS $= 1.35$ g).
   - 🟢 **Stable**: High-speed harmonic pocket ($6375$ RPM, $a_p = 2.2$ mm, RMS $= 0.38$ g).
2. **Machine Bounds Customization**: Tune min/max spindle speed, axis feed, spindle power rating ($5.5$ kW, $7.5$ kW, $11$ kW), and tool chip load limits ($f_{z,min}$ to avoid rubbing, $f_{z,max}$ to prevent breakage).
3. **Live State Diagnosis**: View real-time class classification (`Stable`, `Warning`, `High Risk`), probability distribution bars, and composite chatter risk score.
4. **Baseline vs. Recommended Comparison**: Review side-by-side performance metrics and percentage deltas:
   - Chatter Risk suppression delta
   - Material Removal Rate ($MRR$) delta
   - Estimated cycle time impact
   - Tool wear index
5. **Interactive Tabs**:
   - **Tab 1: Stability Lobe Diagram**: Interactive lobe boundary with baseline and candidate markers.
   - **Tab 2: Vibration Spectrum**: FFT simulated response showing tooth passing harmonics vs structural resonance peak.
   - **Tab 3: Trade-Off Pareto Frontier**: Scatter plot of MRR vs. Chatter Risk.
   - **Tab 4: Candidate Audit & Bound Checks**: Full audit logs detailing which candidates were feasible and exact reasons for any rejected candidates.
   - **Tab 5: ML Explainability**: Hold-out confusion matrix and Gini feature importances.
6. **Data Exports**: Download evaluated candidates CSV, synthetic training data CSV, or executive summary text report.

---



## 📜 License
Developed for academic, research, and hackathon demonstration purposes under the MIT License.
