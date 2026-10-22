# NASA C-MAPSS Sensor Telemetry Reference Guide
## Document ID: NASA-TP-CMAPSS-SENSORS-2024

### Sensor Mapping & Nominal Operational Bounds (FD001 Baseline)

The 21 telemetry channels recorded in the C-MAPSS dataset correspond to specific gas-path stations and mechanical spools:

| Sensor Code | Station / Symbol | Engineering Description | Engineering Units | Degradation Behavior in FD001 |
|---|---|---|---|---|
| `s_1` | T2 | Total temperature at fan inlet | °R (Rankine) | Constant (sea-level test condition: 518.67 °R) |
| `s_2` | T24 | Total temperature at LPC outlet | °R | Drifts upward with compressor degradation (+10 to +15 °R) |
| `s_3` | T30 | Total temperature at HPC outlet | °R | Primary health indicator; climbs steeply before failure |
| `s_4` | T50 | Total temperature at LPT outlet | °R | Exhaust gas temperature; rises significantly with thermal wear |
| `s_5` | P2 | Pressure at fan inlet | psia | Constant (14.696 psia standard sea-level) |
| `s_6` | P15 | Total pressure in bypass-duct | psia | Minimal variance under FD001 operating mode |
| `s_7` | P30 | Total pressure at HPC outlet | psia | Decreases moderately as compressor flow capacity degrades |
| `s_8` | Nf | Physical fan speed | rpm | Drifts upward as FADEC maintains demand |
| `s_9` | Nc | Physical core speed | rpm | Drifts upward to compensate for HPC compression loss |
| `s_10` | epr | Engine pressure ratio (P50/P2) | -- | Constant under FD001 single-regime profile |
| `s_11` | Ps30 | Static pressure at HPC outlet | psia | Increases with internal backpressure as seals wear |
| `s_12` | phi | Ratio of fuel flow to Ps30 | pps/psi | Decreases as fuel metering adjusts for elevated pressure |
| `s_13` | NRf | Corrected fan speed | rpm | Drifts upward with fan erosion |
| `s_14` | NRc | Corrected core speed | rpm | Increases as turbine efficiency degrades |
| `s_15` | BPR | Bypass ratio | -- | Drops as core requires more mass flow to deliver thrust |
| `s_16` | farB | Burner fuel-air ratio | -- | Constant under standard cruise setting in FD001 |
| `s_17` | htBleed | Bleed enthalpy | -- | Climbs steadily; high correlation with HPC thermal wear |
| `s_18` | Nf_dmd | Demanded fan speed | rpm | Commanded setpoint (constant in FD001) |
| `s_19` | PCNfR_dmd | Demanded corrected fan speed | rpm | Commanded setpoint (constant in FD001) |
| `s_20` | W31 | HPT coolant bleed | lbm/s | Drops as bleed valve seals degrade |
| `s_21` | W32 | LPT coolant bleed | lbm/s | Drops with structural and thermal seal leakage |

### Active vs Invariant Sensors in FD001
Notice that sensors `s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, and `s_19` have zero or near-zero variance in FD001 because operating conditions (altitude, Mach number, throttle resolver angle) are held at single sea-level operating condition. The 14 informative channels are:
`s_2`, `s_3`, `s_4`, `s_7`, `s_8`, `s_9`, `s_11`, `s_12`, `s_13`, `s_14`, `s_15`, `s_17`, `s_20`, and `s_21`.
