# NASA C-MAPSS Turbofan Engine Maintenance Manual (Excerpts)
## Document ID: NASA-TP-CMAPSS-MNT-2024
### Section 1: Turbofan Engine Architecture & Gas Path Dynamics

The modular turbofan simulated in the C-MAPSS environment represents a 90,000 lbf thrust class dual-spool commercial aircraft high-bypass turbofan engine. The primary gas path comprises:

1. **Fan (LPC Stage 1)**: Driven by the Low-Pressure Turbine (LPT) via the low-pressure shaft. Regulates inlet total temperature ($T_2$) and inlet pressure ($P_2$).
2. **Low-Pressure Compressor (LPC)**: Multi-stage booster providing initial air compression. Station 24 represents LPC outlet temperature ($T_{24}$).
3. **High-Pressure Compressor (HPC)**: Driven by the High-Pressure Turbine (HPT) via the high-pressure spool. Key indicators include outlet total temperature ($T_{30}$) and static pressure ($P_{s30}$). HPC blade erosion and tip clearance widening cause progressive thermodynamic efficiency losses.
4. **Combustor**: Combines fuel flow with HPC discharge air at fuel-to-air ratio ($farB$) and fuel flow to static pressure ratio ($phi$).
5. **High-Pressure Turbine (HPT)**: Drives the HPC spool under severe thermal conditions. Cooled by compressor discharge bleed air ($W_{31}$).
6. **Low-Pressure Turbine (LPT)**: Drives the fan and LPC spools. Key diagnostic telemetry includes total exhaust temperature ($T_{50}$). Cooled by LPT bleed ($W_{32}$).

### Section 2: Component Degradation Mechanisms

In C-MAPSS FD001 simulations, degradation occurs primarily due to:
- **HPC Flow Capacity & Efficiency Degradation**: As blade roughness increases and tip clearance widens, HPC efficiency drops by up to 1-2%. The engine control system (FADEC) compensates by demanding higher fuel flow ($\phi$) to maintain thrust, elevating HPC outlet temperature ($T_{30}$) and static pressure ($P_{s30}$).
- **Fan & LPC Degradation**: Blade fouling or foreign object debris (FOD) alters corrected fan speed ($NR_f$) and physical fan speed ($N_f$).
- **Turbine Creep and Thermal Degradation**: Elevated turbine operating temperatures cause creep in first-stage HPT nozzle guide vanes. Total temperature at LPT outlet ($T_{50}$) exhibits an upward drift exceeding 15-25 °R over nominal baselines as thermal efficiency decays.
- **Bleed Enthalpy Changes**: Bleed enthalpy ($htBleed$, sensor $s_{17}$) increases linearly with HPC stage deterioration as compression work generates excess heat.

### Section 3: Sensor Drift and Anomaly Classification

Maintenance engineers must rigorously distinguish between **statistical anomalies** and **confirmed mechanical faults**:
- **Statistical Anomaly**: An isolated single-cycle sensor spike exceeding $3\sigma$ from rolling mean caused by transducer noise, electrical interference, or brief flight-level transient adjustments. Does NOT warrant immediate engine teardown unless accompanied by multi-sensor correlation.
- **Confirmed Mechanical Degradation**: Persistent multi-sensor drift sustained over >15 operating cycles (e.g., simultaneous increases in $T_{30}$, $T_{50}$, and $P_{s30}$ coupled with declining bypass ratio $BPR$).
