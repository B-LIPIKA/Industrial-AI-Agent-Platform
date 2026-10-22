# Turbofan Predictive Maintenance & RUL Action Protocols
## Document ID: SOP-AERO-PRD-RUL-2024

### Section 1: Remaining Useful Life (RUL) Threshold Action Grid

The Remaining Useful Life metric in C-MAPSS is defined in discrete operating cycles (flight takeoff-cruise-landing cycles). Maintenance actions must strictly follow this tier classification:

| RUL Tier | Cycle Range | Fleet Status | Mandatory Action Protocol |
|---|---|---|---|
| **CRITICAL / AOG RISK** | $\text{RUL} \le 20\text{ cycles}$ | Grounding Advisory | Immediate boroscope inspection of HPC 3rd-stage blades and HPT nozzle guide vanes. Schedule engine shop visit within 48 operating hours. Replace worn compressor stages. |
| **HIGH RISK** | $20 < \text{RUL} \le 45\text{ cycles}$ | High Urgency | Restrict engine from extended-range twin-engine operations (ETOPS). Schedule line maintenance inspection at primary hub within 5 calendar days. |
| **MODERATE RISK** | $45 < \text{RUL} \le 80\text{ cycles}$ | Monitor | Increase automated telemetry sampling frequency. Flag for review during next A-check. Inspect oil debris monitors and bleed valves. |
| **NOMINAL** | $\text{RUL} > 80\text{ cycles}$ | Normal Operations | Engine operating within nominal fleet degradation envelope. Standard scheduled visual inspection per maintenance cycle plan. |

### Section 2: Conversion of Operating Cycles to Operational Time

- **Important Limitation**: RUL in C-MAPSS measures **discrete operational cycles**, not calendar days or running engine hours.
- A single cycle in commercial operations represents one complete mission: taxi, takeoff, climb, cruise, descent, thrust reverser deployment, and shutdown.
- For typical short-haul commercial fleets, 1 cycle $\approx$ 1.5 to 2.5 flight hours. In long-haul wide-body operations, 1 cycle may exceed 8 to 12 flight hours.
- Conversion to calendar maintenance time requires:
  $$\text{Calendar Days} = \frac{\text{RUL (cycles)}}{\text{Average Daily Utilization (cycles/day)}}$$
- Thermal cycle fatigue accumulates non-linearly during maximum takeoff thrust transitions ($N_1$, $N_2$ spool accelerations).

### Section 3: Isolation Forest & Statistical Anomaly Rules

- Telemetry points exhibiting an Isolation Forest anomaly score $< -0.10$ or rolling $Z$-score exceeding $\pm 2.5\sigma$ over a 15-cycle window are tagged as candidate anomalies.
- If fewer than 2 distinct sensor channels exhibit simultaneous anomalies, classify the event as **transient sensor noise**.
- If 3 or more degradation-sensitive sensors (`s_2`, `s_3`, `s_4`, `s_7`, `s_11`, `s_17`) deviate concurrently, escalate to an **active subsystem degradation alert**.
