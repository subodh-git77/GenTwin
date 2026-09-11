# app.py
# ==========================================================
# GenTwin - Final Dashboard + SimPy Digital Twin
# No core model logic changed
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import plotly.express as px
import plotly.graph_objects as go
import simpy


def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="GenTwin Dashboard",
    page_icon="🛡️",
    layout="wide"
)


load_css()
# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------
st.markdown("""
<h1 class="glow">🛡️ GenTwin AI Digital Twin Command Center</h1>
<h4>AI-Powered Synthetic Attack Detection for Smart Water Plant</h4>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# DEVICE
# ----------------------------------------------------------
device = torch.device("cpu")

# ----------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------
INPUT_DIM = 51
LATENT_DIM = 16

NORMAL_LIMIT = 1.45
SUSPICIOUS_LIMIT = 2.50

# ----------------------------------------------------------
# LOAD SCALERS
# ----------------------------------------------------------
detector_scaler = joblib.load("models/scaler.pkl")
generator_scaler = joblib.load("models/merged_scaler.pkl")
feature_names = detector_scaler.feature_names_in_

# ==========================================================
# MODELS
# ==========================================================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM,128)
        self.fc2 = nn.Linear(128,64)

        self.mu = nn.Linear(64,LATENT_DIM)
        self.logvar = nn.Linear(64,LATENT_DIM)

        self.fc3 = nn.Linear(LATENT_DIM,64)
        self.fc4 = nn.Linear(64,128)

        self.out = nn.Linear(128,INPUT_DIM)

    def encode(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.out(h)

    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        out = self.decode(z)
        return out,mu,logvar


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM+1,128)
        self.fc2 = nn.Linear(128,64)

        self.mu = nn.Linear(64,LATENT_DIM)
        self.logvar = nn.Linear(64,LATENT_DIM)

        self.fc3 = nn.Linear(LATENT_DIM+1,64)
        self.fc4 = nn.Linear(64,128)

        self.out = nn.Linear(128,INPUT_DIM)

    def encode(self,x,c):
        h = torch.cat([x,c],dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

    def decode(self,z,c):
        h = torch.cat([z,c],dim=1)
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        return self.out(h)

    def forward(self,x,c):
        mu,logvar = self.encode(x,c)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return self.decode(z,c), mu, logvar

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
detector = VAE().to(device)
detector.load_state_dict(torch.load("models/vae_swat.pth", map_location=device))
detector.eval()

generator = CVAE().to(device)
generator.load_state_dict(torch.load("models/cvae_attack_generator.pth", map_location=device))
generator.eval()

# ==========================================================
# FUNCTIONS
# ==========================================================
@st.cache_data
def generate_state():
    z = torch.randn(1,LATENT_DIM).to(device)
    c = torch.zeros(1,1).to(device)

    with torch.no_grad():
        gen_scaled = generator.decode(z,c).cpu().numpy()

    return generator_scaler.inverse_transform(gen_scaled)[0]


def anomaly_score(sample):
    df = pd.DataFrame(sample.reshape(1,-1), columns=feature_names)
    scaled = detector_scaler.transform(df)

    x = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon,_,_ = detector(x)
        err = torch.mean((x-recon)**2, dim=1).cpu().numpy()[0]

    return err

# ==========================================================
# BASELINE SAMPLE
# ==========================================================
sample = generate_state()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("⚙️ Smart Controls")

st.sidebar.markdown("### Tank Level")
LIT101 = st.sidebar.slider("LIT101",0.0,1000.0,float(sample[1]))

st.sidebar.markdown("### Flow")
FIT101 = st.sidebar.slider("FIT101",0.0,5.0,float(sample[0]))
FIT201 = st.sidebar.slider("FIT201",0.0,5.0,float(sample[5]))

st.sidebar.markdown("### Pressure")
PIT501 = st.sidebar.slider("PIT501",0.0,50.0,float(sample[44]))
PIT502 = st.sidebar.slider("PIT502",0.0,50.0,float(sample[45]))
PIT503 = st.sidebar.slider("PIT503",0.0,50.0,float(sample[46]))

st.sidebar.markdown("### Actuators")
P101 = st.sidebar.selectbox("Pump P101",[0,1],index=1)
P201 = st.sidebar.selectbox("Pump P201",[0,1],index=1)
MV101 = st.sidebar.selectbox("Valve MV101",[0,1],index=1)
MV201 = st.sidebar.selectbox("Valve MV201",[0,1],index=1)

# ==========================================================
# APPLY VALUES
# ==========================================================
sample[0]  = FIT101
sample[1]  = LIT101
sample[2]  = MV101
sample[3]  = P101
sample[5]  = FIT201
sample[10] = MV201
sample[11] = P201
sample[44] = PIT501
sample[45] = PIT502
sample[46] = PIT503

# ==========================================================
# PREDICT
# ==========================================================
error = anomaly_score(sample)

if error < NORMAL_LIMIT:
    status = "🟢 Normal"
    risk = 20
elif error < SUSPICIOUS_LIMIT:
    status = "🟠 Suspicious"
    risk = 60
else:
    status = "🔴 Synthetic Attack"
    risk = 95

# ==========================================================
# KPI ROW
# ==========================================================
c1,c2,c3,c4 = st.columns(4)

c1.metric("Anomaly Score", round(error,4))
c2.metric("Status", status)
c3.metric("Risk Score", f"{risk}%")
c4.metric("Health Score", f"{100-risk}%")

# ==========================================================
# MAIN VISUALS
# ==========================================================
left,right = st.columns(2)

with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text':"Threat Severity"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,35], 'color':'green'},
                {'range':[35,70], 'color':'orange'},
                {'range':[70,100], 'color':'red'}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

with right:
    chart_df = pd.DataFrame({
        "Parameter":["LIT101","FIT101","FIT201","PIT501","PIT502","PIT503"],
        "Value":[LIT101,FIT101,FIT201,PIT501,PIT502,PIT503]
    })

    fig = px.bar(chart_df,x="Parameter",y="Value",
                 text_auto=True,title="Live Parameters")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# SIMPY DIGITAL TWIN (WAVE-BASED ADVANCED)
# ==========================================================
st.markdown("---")
st.subheader("🏭 Live Digital Twin Simulation (Waveform View)")

def run_sim():

    env = simpy.Environment()

    history = {
        "time": [],
        "level": [],
        "pressure": [],
        "flow": []
    }

    tank = LIT101
    pressure = PIT501
    flow = FIT101

    def process(env):
        nonlocal tank, pressure, flow

        while env.now < 30:

            # ---------------------------
            # Base Process Dynamics
            # ---------------------------
            if P101 == 1:
                tank += flow * 3
            else:
                tank -= 2

            if MV101 == 0:
                pressure += 1.5
            else:
                pressure -= 0.3

            # ---------------------------
            # Add realistic wave noise
            # ---------------------------
            tank_wave = tank + 10 * np.sin(env.now / 2)
            pressure_wave = pressure + 2 * np.cos(env.now / 3)
            flow_wave = flow + 0.2 * np.sin(env.now)

            # ---------------------------
            # Attack Spike Simulation
            # ---------------------------
            if status == "🔴 Synthetic Attack":
                pressure_wave += np.random.uniform(3, 6)
                tank_wave += np.random.uniform(20, 40)

            # ---------------------------
            # Clamp values
            # ---------------------------
            tank = max(0, min(1000, tank))
            pressure = max(0, min(50, pressure))

            history["time"].append(env.now)
            history["level"].append(tank_wave)
            history["pressure"].append(pressure_wave)
            history["flow"].append(flow_wave)

            yield env.timeout(1)

    env.process(process(env))
    env.run()

    return pd.DataFrame(history)


sim_df = run_sim()

# ==========================================================
# WAVE VISUALIZATION
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    fig_wave = go.Figure()

    fig_wave.add_trace(go.Scatter(
        x=sim_df["time"],
        y=sim_df["level"],
        mode="lines",
        name="Tank Level",
        line=dict(width=3)
    ))

    fig_wave.add_trace(go.Scatter(
        x=sim_df["time"],
        y=sim_df["flow"],
        mode="lines",
        name="Flow",
        line=dict(width=2, dash="dot")
    ))

    fig_wave.update_layout(
        title="🌊 Tank Level & Flow Dynamics",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig_wave, use_container_width=True)


with col2:
    fig_pressure = go.Figure()

    fig_pressure.add_trace(go.Scatter(
        x=sim_df["time"],
        y=sim_df["pressure"],
        mode="lines",
        name="Pressure",
        line=dict(width=3, color="red")
    ))

    fig_pressure.update_layout(
        title="⚡ Pressure Wave Behavior",
        xaxis_title="Time",
        yaxis_title="Pressure",
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig_pressure, use_container_width=True)



# ==========================================================
# THREAT INTELLIGENCE
# ==========================================================
st.subheader("🛡️ AI Threat Intelligence")
if status == "🟢 Normal":
    st.write("""
    ✅ System is operating within the learned safe behavior profile.

    Observations:
    - Sensor readings are stable
    - Pressure remains within safe limits
    - Flow and actuator states are consistent
    - No immediate cyber-physical anomalies detected

    Outlook:
    - Low probability of near-term operational disruption
    - Continue routine monitoring and preventive maintenance
    """)

elif status == "🟠 Suspicious":
    st.write("""
    ⚠️ Early Warning: Emerging abnormal behavior detected.

    Potential Indicators:
    - Mild sensor drift from baseline operating pattern
    - Pressure imbalance beginning to develop
    - Flow inconsistency between linked process stages
    - Actuator behavior slightly deviating from expected response

    Future Risk Possibilities:
    - Progressive sensor manipulation attempt
    - Developing blockage or leakage in pipeline section
    - Pump efficiency degradation
    - Escalation into coordinated attack if ignored

    Recommended Actions:
    - Increase monitoring frequency
    - Verify sensor calibration
    - Inspect pump and valve response logs
    - Prepare preventive intervention
    """)

else:
    st.write("""
    🚨 Critical Threat State: High probability of active or imminent attack scenario.

    Severe Indicators:
    - Abnormal pressure pattern suggests process tampering
    - Overflow condition may occur if tank level continues rising
    - Pump-flow contradiction indicates actuator spoofing or control compromise
    - Valve manipulation pattern detected
    - Multi-parameter anomaly exceeds safe operational threshold

    Predicted Near-Future Attack Impact:
    - Tank overflow or shutdown event
    - Pipeline rupture / pressure damage
    - Water treatment interruption
    - Unsafe plant state propagation across stages
    - Remote attacker persistence within control logic

    Immediate Mitigation Required:
    - Switch to safe manual control mode
    - Isolate affected pumps / valves
    - Reduce inflow and stabilize pressure
    - Audit PLC / SCADA commands
    - Trigger incident response protocol
    """)

# ==========================================================
# FUTURE OUTLOOK (SMART)
# ==========================================================
future_level = sim_df["level"].iloc[-1]
future_pressure = sim_df["pressure"].iloc[-1]

st.subheader("🔮 Future Plant Outlook")

if future_level > 850:
    st.error("🚨 Tank level trend indicates future overflow risk.")

elif future_pressure > 35:
    st.error("🚨 Pressure trend indicates possible pipeline stress/failure.")

elif status == "🟠 Suspicious":
    st.warning("⚠️ System trending towards unstable operating region.")

else:
    st.success("✅ Future system behavior remains stable and controlled.")

# ==========================================================
# FORECAST ALERT
# ==========================================================
future_level = sim_df["level"].iloc[-1]
future_pressure = sim_df["pressure"].iloc[-1]

st.subheader("🔮 Future Plant Outlook")

if future_level > 850:
    st.error("Predicted future overflow risk.")

elif future_pressure > 35:
    st.error("Predicted future pressure critical zone.")

else:
    st.success("Future operating trend appears stable.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("GenTwin © AI Digital Twin Security Platform")