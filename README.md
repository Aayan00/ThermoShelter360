# ThermoShelter360 🏔️⚡

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-5.18+-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/pytest-13%20passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ThermoShelter360** is a physics-grounded, software-based thermal simulation, 3D architectural modeling, and optimization platform designed specifically for **area-specific passive solar shelters in extreme cold high-altitude regions** such as Ladakh (India), Spiti, and the Trans-Himalayan plateau.

---

## 🌟 Key Features

- 🏛️ **Interactive 3D Parametric CAD Engine**: Real-time 3D shelter mesh with realistic material textures, south-facing solar glazing, and customizable 3D incident solar radiation vectors.
- 🏰 **Architectural Gallery of 6 Passive Himalayan Shelters**: Includes pre-configured Ladakh Trombe Wall Shelters, Military Arctic Pods, Spiti Earthbag Cabins, Tibetan Stone Refuges, Geodesic Polar Domes, and Solar Greenhouse Habitats with one-click simulator application.
- 🎬 **3 Dynamic Animated Visualizations**:
  1. *24-Hour Diurnal Temperature Wave Animation* (Hourly progression with comfort status indicator)
  2. *24-Hour Solar Sun Path Trajectory* (Sun elevation angles and incident irradiance)
  3. *Dynamic Envelope Heat Flux Balance* (Hourly solar gains vs component conduction losses)
- ⚡ **First-Principle Lumped-Capacitance Thermal Engine**: Forward Euler transient energy balance incorporating multi-layer composite wall assemblies (e.g. Adobe + Sheep Wool, Brick + EPS), fenestrations, roof, door, and high-altitude ventilation losses ($\rho \approx 1.05\text{ kg/m}^3$).
- 🧱 **Multi-Material Benchmarking**: Automated comparison across traditional vernacular and modern insulated materials with neutral indicators (*"Lowest heat loss"*, *"Highest average indoor temp"*, *"Lowest supplemental heating demand"*).
- 🔬 **User-Imported ANSYS CFD Benchmark Validation**: Statistical telemetry computing Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and max discrepancy with overlay charts.
- 📥 **Multi-Format Engineering Exports**: Direct download of Simulation CSVs, Comparison CSVs, multi-worksheet Microsoft Excel workbooks (`.xlsx`), and executive PDF summary reports (`.pdf`).
- 💾 **SQLite Simulation Database**: Integrated persistence layer for tracking, comparing, and managing historical shelter simulation runs.

---

## 📸 Interface & Architectural Showcase

```
+---------------------------------------------------------------------------------------+
|  🏔️ ThermoShelter360 — High-Altitude Passive Solar Shelter Simulation Platform         |
+---------------------------------------------------------------------------------------+
|  [3D Shelter Visualizer] [Architecture Gallery] [Animations] [Simulation KPIs] [ANSYS]|
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   🏛️ 3D PARAMETRIC SHELTER MODEL                  📊 24-HR DIURNAL TEMPERATURE PROFILE|
|   +-------------------------------+               +-------------------------------+   |
|   |         /\  Gable Roof        |   ☀️ Sun Beam | 25°C|       [Comfort 18-24°C] |   |
|   |        /  \                   |  ===> (35°)   | 20°C|~~~~~~~~/~~~~~~~~~\~~~~~~|   |
|   |       /____\                  |               | 15°C|---T_in------------------|   |
|   |      |  []  | (South Glazing) |               |  0°C|                         |   |
|   |      |  __  |                 |               |-20°C|_______T_out____________ |   |
|   |      |_|__|_| (Ground Plane)  |               +-------------------------------+   |
|   +-------------------------------+                00:00        12:00        24:00    |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

---

## 🏰 Architectural Gallery Presets

| Shelter Design | Vernacular / Structural Type | Key Materials & Insulation | Best Suited Elevation |
| :--- | :--- | :--- | :--- |
| **Ladakh Vernacular Trombe Shelter** | Indirect Solar Gain Trombe Wall | 450mm Adobe Earth + South Glazing | 3,500m+ ASL (Leh, Changthang) |
| **High-Altitude Military Defense Pod** | Super-Insulated Prefabricated Pod | Composite Panel + 100mm EPS Core | 4,800m+ ASL (Siachen, Kargil) |
| **Spiti Super-Insulated Earthbag Cabin** | Eco Earthbag Sunspace Habitat | Earthbag Mass + 50mm Sheep Wool | 3,800m+ ASL (Kaza, Spiti) |
| **Tibetan High-Mass Stone Refuge** | Granitic Heavy Thermal Mass | 500mm Stone + 60mm Mineral Wool | 4,000m+ ASL (Zanskar Range) |
| **Geodesic Cold-Arid Polar Dome** | Aerodynamic Low-Surface Dome | Timber Struts + 80mm EPS Triangles | 4,300m+ ASL (Pangong Tso Basin) |
| **Solar Greenhouse Habitat** | Solar Greenhouse & Living Core | 400mm Rammed Earth + Straw Board | 3,200m+ ASL (Nubra Valley) |

---

## 📐 Physics & Mathematical Governing Equations

### 1. Transient Lumped Capacitance Energy Balance
$$\mathbf{C_{th} \cdot \frac{dT_{in}}{dt} = Q_{solar}(t) + Q_{internal} - Q_{loss}(t)}$$

Discretized using forward Euler numerical integration:
$$T_{in}(t + \Delta t) = T_{in}(t) + \frac{\Delta t}{C_{th}} \left( Q_{solar}(t) + Q_{internal} - Q_{loss}(t) \right)$$

Where:
- $T_{in}(t)$ = Indoor dry-bulb temperature ($^\circ\text{C}$)
- $C_{th}$ = Effective lumped thermal capacitance of building envelope and air ($J/K$)
- $\Delta t$ = Simulation timestep ($s$)
- $Q_{solar}(t)$ = Incident solar heat gain through apertures ($W$)
- $Q_{internal}$ = Metabolic and internal appliance heat load ($W$)
- $Q_{loss}(t)$ = Total envelope and ventilation heat loss ($W$)

---

### 2. Envelope Heat Loss Formulations

#### Single & Multi-Layer Composite Wall Conduction
$$R_{total} = \sum_{i=1}^{n} \frac{d_i}{k_i}, \quad U_{wall} = \frac{1}{R_{total}}$$
$$Q_{wall} = U_{wall} \cdot A_{net} \cdot (T_{in} - T_{out})$$

#### Roof, Glazing, and Door Transmission Losses
$$Q_{roof} = U_{roof} \cdot A_{roof} \cdot (T_{in} - T_{out})$$
$$Q_{window} = U_{window} \cdot A_{window} \cdot (T_{in} - T_{out})$$
$$Q_{door} = U_{door} \cdot A_{door} \cdot (T_{in} - T_{out})$$

#### High-Altitude Ventilation & Infiltration Loss
$$Q_{vent} = \rho_{air} \cdot c_{p, air} \cdot \dot{V} \cdot (T_{in} - T_{out})$$
Where $\rho_{air} \approx 1.05\text{ kg/m}^3$ at 3,500m elevation in Leh, Ladakh, and $c_{p, air} = 1005\text{ J}/(\text{kg}\cdot\text{K})$.

---

### 3. Solar Radiation Model
$$Q_{solar} = \alpha \cdot G \cdot A_{aperture}$$
Where $\alpha$ is the surface solar absorptivity coefficient ($0.0 \le \alpha \le 1.0$), $G$ is global surface solar irradiance ($\text{W/m}^2$), and $A_{aperture}$ is the effective solar aperture area ($\text{m}^2$).

---

### 4. Thermal Comfort Evaluation
Indoor temperature is evaluated against the **ASHRAE / ISO 7730 18°C–24°C Comfort Band**:
- **Cold**: $T_{in} < 18^\circ\text{C}$
- **Comfortable**: $18^\circ\text{C} \le T_{in} \le 24^\circ\text{C}$
- **Hot**: $T_{in} > 24^\circ\text{C}$

Supplemental heating demand ($E_{heat}$ in $\text{kWh}$):
$$E_{heat} = \sum_{t} \frac{\max(0, 18.0 - T_{in}(t)) \cdot C_{th}}{3.6 \times 10^6}$$

---

## 💻 Installation & Quick Start

### 1. Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone & Install
```bash
git clone https://github.com/your-org/ThermoShelter360.git
cd ThermoShelter360
pip install -r requirements.txt
```

### 3. Run Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 4. Run Automated Test Suite
```bash
pytest -v tests/
```

---

## 🔍 Assumptions & Limitations

1. **Lumped Thermal Capacitance**: The interior air volume is assumed to be well-mixed with uniform spatial temperature.
2. **Quasi-Steady Conduction**: Heat transmission through envelope components is modeled using discrete 1D thermal resistance networks per timestep.
3. **User-Imported ANSYS Benchmark Validation**: The validation module compares user-imported benchmark CSV datasets; it does not launch proprietary external solvers.
4. **Air Density Adaptation**: High-altitude air density default is set to $1.05\text{ kg/m}^3$ (corresponding to ~3,500m altitude in Ladakh).

---

## 📚 References & Standards

1. **ASHRAE Handbook of Fundamentals** (2021) — *Chapter 14: Climatic Design Information & Chapter 18: Nonresidential Cooling and Heating Load Calculations*.
2. **ISO 13786:2017** — *Thermal performance of building components — Dynamic thermal characteristics — Calculation methods*.
3. **ISO 7730:2005** — *Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices*.
4. **EnergyPlus™ Engineering Reference** (U.S. Department of Energy) — *Building Envelope and Conduction Transfer Functions*.
5. **Druk White Lotus School & SECMOL Passive Solar Architecture Studies** — *Vernacular and passive solar building techniques in Ladakh, India*.

---

## 📜 License
ThermoShelter360 is open-source under the MIT License.
