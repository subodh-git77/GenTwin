# 🛡️ GenTwin — AI Digital Twin for Cyber-Physical Attack Detection

**GenTwin** is an interactive Streamlit dashboard that combines **generative deep learning** with a **SimPy-based digital twin simulation** to detect, generate, and visualize cyberattacks on a smart water treatment plant. It's built around the SWaT (Secure Water Treatment) industrial control system dataset and demonstrates how anomaly-detection models and synthetic-data generators can work together to build a real-time security "command center" for critical infrastructure.

🔗 **Live demo:** [gentwin-uttam-maurya.streamlit.app](https://gentwin-uttam-maurya.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 What It Does

Industrial control systems (ICS) like water treatment plants are increasingly targeted by cyber-physical attacks that manipulate sensors and actuators (pumps, valves, tanks) to cause real-world damage. GenTwin tackles two sides of this problem in one app:

1. **Detects anomalies** in real time using a **Variational Autoencoder (VAE)** trained on normal plant behavior — high reconstruction error signals a potential attack.
2. **Generates synthetic attack scenarios** using a **Conditional VAE (CVAE)**, useful for stress-testing detection systems and building richer training data.
3. **Simulates the physical plant** with **SimPy**, turning sensor values into a live, animated digital twin — so you can *see* how a tank level, flow rate, or pressure spike propagates over time.
4. **Visualizes risk** with an interactive gauge, live parameter charts, and rule-based threat intelligence reports (Normal / Suspicious / Attack), plus a short-term outlook forecast.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **VAE Anomaly Detector** | Reconstructs 51 SWaT sensor/actuator features; flags deviations via reconstruction error |
| 🧪 **CVAE Attack Generator** | Generates realistic synthetic plant states (normal or attack-conditioned) for testing |
| 🏭 **SimPy Digital Twin** | Discrete-event simulation of tank level, pressure, and flow dynamics with waveform noise |
| 🎛️ **Interactive Controls** | Live sliders for tank level, flow, and pressure + toggles for pumps (P101/P201) and valves (MV101/MV201) |
| 📊 **Live Dashboards** | Plotly gauges, bar charts, and dual waveform plots (tank/flow & pressure) |
| 🚨 **Threat Intelligence Engine** | Rule-based narrative reports explaining *why* a state is Normal, Suspicious, or an Attack, with recommended actions |
| 🔮 **Future Outlook** | Projects near-term plant trajectory (overflow risk, pressure stress) from simulation trends |

---

## 🏗️ How It Works

```
                ┌─────────────────────┐
                │   CVAE Generator     │  ── generates a baseline plant state
                │ (cvae_attack_generator.pth) │
                └──────────┬──────────┘
                           │
                 sample (51 features)
                           │
        ┌──────────────────┴───────────────────┐
        │   User adjusts via sidebar sliders     │
        │  (tank level, flow, pressure, pumps,   │
        │           valves)                      │
        └──────────────────┬───────────────────┘
                           │
                ┌──────────▼──────────┐
                │    VAE Detector      │  ── scores reconstruction error
                │   (vae_swat.pth)     │
                └──────────┬──────────┘
                           │
              error < 1.45 → 🟢 Normal
        1.45 ≤ error < 2.50 → 🟠 Suspicious
              error ≥ 2.50 → 🔴 Synthetic Attack
                           │
                ┌──────────▼──────────┐
                │   SimPy Digital Twin  │  ── simulates 30 time-steps of
                │                       │     tank/pressure/flow dynamics,
                │                       │     injecting attack spikes if
                │                       │     status = Attack
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Dashboard & Threat   │
                │  Intelligence Report  │
                └───────────────────────┘
```

### Model Architecture

Both models operate on **51 SWaT process features** (flow, level, pressure, conductivity, and pump/valve states across all 6 process stages, P1–P6).

- **VAE (Detector):** `51 → 128 → 64 → z(16) → 64 → 128 → 51`, trained on normal operating data only. Anomaly score = mean squared reconstruction error.
- **CVAE (Generator):** Same backbone, conditioned on a binary label `c` (normal/attack), enabling controllable synthetic sample generation via `models/synthetic_attacks.csv`.

---

## 📁 Project Structure

```
GenTwin-main/
├── app.py                          # Main Streamlit application
├── assets/
│   └── style.css                   # Custom dashboard theming (green/blue gradient UI)
├── models/
│   ├── vae_swat.pth                # Trained VAE anomaly detector weights
│   ├── cvae_attack_generator.pth   # Trained CVAE synthetic attack generator weights
│   ├── scaler.pkl                  # Feature scaler for the detector
│   ├── merged_scaler.pkl           # Feature scaler for the generator
│   └── synthetic_attacks.csv       # Pre-generated synthetic attack samples (5,000 rows)
├── .devcontainer/
│   └── devcontainer.json           # GitHub Codespaces / VS Code dev container config
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🖥️ Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/subodh-git77/GenTwin.git
cd GenTwin
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

> 💡 **Codespaces:** This repo includes a `.devcontainer` config — open it in GitHub Codespaces and the app will launch automatically on port `8501`.

### Dependencies

```
streamlit
pandas
numpy
plotly
torch
scikit-learn
joblib
simpy
```

See [`requirements.txt`](./requirements.txt) for pinned versions.

---

## 🎮 Using the Dashboard

1. **Adjust sensors** in the sidebar — tank level, flow rates (FIT101/FIT201), pressure readings (PIT501–503), and toggle pumps/valves.
2. **Watch the KPI row** update instantly: anomaly score, status, risk %, and health %.
3. **Check the gauge and bar chart** for an at-a-glance threat severity view.
4. **Scroll to the digital twin** to see a live 30-step waveform simulation of how the plant would physically respond — including attack-induced spikes if the current state is flagged as malicious.
5. **Read the AI Threat Intelligence report** for a plain-language explanation of what's happening and recommended mitigation steps.
6. **Check the Future Outlook** section for a short-term forecast (e.g., overflow or pressure-failure risk).

---

## 🔬 Dataset & Context

This project is built around **SWaT (Secure Water Treatment)**, a widely used ICS security research testbed dataset (iTrust, Singapore University of Technology and Design) that captures sensor/actuator telemetry from a scaled-down water treatment plant under both normal operation and various physical/cyber attack scenarios. GenTwin uses this feature schema (51 columns spanning process stages P1–P6) to train its detection and generation models.

---

## 🧭 Roadmap Ideas

- [ ] Swap the rule-based threat intelligence text with an LLM-generated dynamic report
- [ ] Add historical time-series playback instead of single-snapshot analysis
- [ ] Support live streaming/replay of full SWaT attack sequences
- [ ] Add model retraining UI for custom plant configurations
- [ ] Export incident reports as PDF

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues) or open a pull request.

---

## 📜 License

This project is open source. Add your preferred license (e.g., MIT) in a `LICENSE` file if not already present.

---

## 🙋 Team

GenTwin is a **group project** led by **Uttam Maurya**, along with the contributing team members.

🔗 [Live App](https://gentwin-uttam-maurya.streamlit.app/)

> Add the rest of the team's names here, e.g.:
> - Subodh Kumar Agrahari
> - Upendra Singh
> - Vanshita Agrawal
> - Vikas Kumar Dhawan
---

<p align="center"><i>GenTwin © AI Digital Twin Security Platform</i></p>
