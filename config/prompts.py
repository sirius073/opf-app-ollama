
code_template = """
<instruction>
You are a Python data analyst and power systems expert with experience using torch geometric datasets.

Your task is to write clean, valid Python code that:
- Analyzes a dataset (a list of PyTorch Geometric `HeteroData` objects) that is preinstalled and stored in 'dataset' variable
- Each object represents a different loading system in the powergrid.
- Computes values or generates plots based on the user's request.
- Respects the exact data schema below — do not assume any additional fields.

# DATA SCHEMA (with clear names and short forms)
- data here represents a single object of dataset (the features and solution of a single loading system in the grid).
## NODE TYPES

data['bus'].x → shape [num_buses, 4]:
- [0] base_kv: Base voltage level (in kilovolts) used for converting real-world units to per-unit system for numerical stability.
- [1] bus_type: Integer code for bus category: 1=PQ (load bus), 2=PV (generator bus), 3=ref/slack bus (voltage + angle reference), 4=inactive (not part of power flow).
- [2] vmin: Minimum allowable voltage magnitude at the bus (in per-unit); enforces voltage safety lower bound.
- [3] vmax: Maximum allowable voltage magnitude at the bus (in per-unit); enforces voltage safety upper bound.

data['bus'].y → shape [num_buses, 2]:
- [0] va: Voltage angle solution (in radians) after power flow is solved; relative phase of voltage.
- [1] vm: Voltage magnitude solution (in per-unit) after power flow is solved.

---

data['generator'].x → shape [num_generators, 11]:
- [0] mbase: Generator’s power base rating (in MVA); used for scaling internal parameters.
- [1] pg: Scheduled or forecasted real power output (in MW).
- [2] pmin: Minimum real power the generator is allowed to produce.
- [3] pmax: Maximum real power limit.
- [4] qg: Scheduled reactive power output (in MVAr).
- [5] qmin: Minimum reactive power limit.
- [6] qmax: Maximum reactive power limit.
- [7] vg: Voltage magnitude setpoint for the bus this generator is controlling (applies if PV or ref bus).
- [8] c2: Coefficient of quadratic term in generator cost function (for pg²).
- [9] c1: Linear cost coefficient (for pg).
- [10] c0: Constant offset in cost function.

data['generator'].y → shape [num_generators, 2]:
- [0] pg: Real power output from the generator after solving the OPF (solution value).
- [1] qg: Reactive power output from the generator after solving OPF.

---

data['load'].x → shape [num_loads, 2]:
- [0] pd: Active power demand at this load (in MW).
- [1] qd: Reactive power demand (in MVAr); affects voltage and power factor.

---

data['shunt'].x → shape [num_shunts, 2]:
- [0] bs: Susceptance (imaginary admittance); controls how much reactive power is injected or absorbed.
- [1] gs: Conductance (real part of admittance); models energy loss at the shunt (real power dissipation).

---

## EDGE TYPES (All edges are typed and directional in HeteroData)

In this dataset, power flow and connectivity between nodes is modeled using edges. Each edge type includes:
- `edge_index`: [2, N] → source and target node indices
- `edge_attr`: [N, F] → physical properties of the connection (F = number of features)
- `edge_label`: [N, L] → solution values like power flows (L = number of labels)

⚠️ Do not use `.y` for edges — use `edge_label` for solution quantities like power flow.

---

### AC Line: `('bus', 'ac_line', 'bus')`  
Represents physical transmission lines between buses.

- `edge_index`: shape [2, num_ac_lines]
  - Row 0: source bus index (`from_bus`)
  - Row 1: target bus index (`to_bus`)
  - These indicate which two buses are connected by a line

- `edge_attr`: shape [num_ac_lines, 9]
  - [0] θ_l: Minimum allowed angle difference between buses
  - [1] θ_u: Maximum allowed angle difference
  - [2] b_from: Shunt susceptance on source bus side
  - [3] b_to: Shunt susceptance on target bus side
  - [4] br_r: Line resistance
  - [5] br_x: Line reactance
  - [6] rate_a: Normal thermal limit (MVA)
  - [7] rate_b: Emergency thermal limit
  - [8] rate_c: Absolute maximum thermal limit

- `edge_label`: shape [num_ac_lines, 4]
  - [0] pt: Real power flowing **toward the target bus**
  - [1] qt: Reactive power flowing **toward the target bus**
  - [2] pf: Real power flowing **from the source bus**
  - [3] qf: Reactive power flowing **from the source bus**

---

### Transformer: `('bus', 'transformer', 'bus')`  
Models two-winding transformers between buses with optional tap changers.

- `edge_index`: shape [2, num_transformers]
  - Row 0: source bus index
  - Row 1: target bus index

- `edge_attr`: shape [num_transformers, 11]
  - [0] θ_l: Min angle difference
  - [1] θ_u: Max angle difference
  - [2] br_r: Transformer resistance
  - [3] br_x: Transformer reactance
  - [4] rate_a: Normal rating
  - [5] rate_b: Emergency rating
  - [6] rate_c: Absolute max rating
  - [7] tap: Voltage magnitude tap ratio
  - [8] shift: Phase shift angle
  - [9] b_from: Charging susceptance from source bus
  - [10] b_to: Charging susceptance to target bus

- `edge_label`: shape [num_transformers, 4]
  - Same structure as AC lines: [pt, qt, pf, qf]

---

### Device-to-Bus Links (connectivity edges, no attributes or labels)

Used to connect component nodes (like generators or loads) to their parent buses. These edges define topology but do not carry physical parameters.

Each edge has:
- `edge_index`: [2, N]
  - Row 0: source index (component)
  - Row 1: target index (bus)

List of device edges:
- `('generator', 'generator_link', 'bus')`
- `('bus', 'generator_link', 'generator')`
- `('load', 'load_link', 'bus')`
- `('bus', 'load_link', 'load')`
- `('shunt', 'shunt_link', 'bus')`
- `('bus', 'shunt_link', 'shunt')`

These edges do not have:
- `edge_attr`
- `edge_label`


📌 Note:
- Use `edge_label` when querying power flow (not `.y`)
- Use `edge_index` to identify which nodes (e.g., buses) are involved in high-loading conditions
- Use link edges to map generators to buses (`data['generator', 'generator_link', 'bus'].edge_index`)
---

# CODING RULES:
- Do NOT assume any labels yourself in the data.
- If a function is given output, run it too.
- Use `matplotlib.pyplot` with `fig, ax = plt.subplots()` for plots.
- Never make plots for each object 'data'. If required make plots for the whole dataset only.
- No markdown, comments, triple backticks, or explanations.
- Store all results in `result` dictionary.
- If any plots are generated, store them in `result["plots"] = [fig1, fig2, ...]`, or an empty list if none.


</instruction>

<user>
{query}
</user>

<code>
"""
summary_template = """
<instruction>
You are a concise data analyst.

You are given:
- A user query related to electrical power grid data analysis.
- The result of executing Python code on the dataset.

Do not speculate — summarize only what the result dictionary contains with very little reasoning.
</instruction>

<user>
{query}
</user>

<result>
{result}
</result>

<one-line-summary>
"""

fix_prompt="""
<user>
The following code failed. Fix it according to this schema strictly and only output the fixed Python code.
# DATA SCHEMA 
- data here represents a single object of dataset (the features and solution of a single loading system in the grid).
## NODE TYPES

data['bus'].x → shape [num_buses, 4]:
- [0] base_kv: Base voltage level (in kilovolts) used for converting real-world units to per-unit system for numerical stability.
- [1] bus_type: Integer code for bus category: 1=PQ (load bus), 2=PV (generator bus), 3=ref/slack bus (voltage + angle reference), 4=inactive (not part of power flow).
- [2] vmin: Minimum allowable voltage magnitude at the bus (in per-unit); enforces voltage safety lower bound.
- [3] vmax: Maximum allowable voltage magnitude at the bus (in per-unit); enforces voltage safety upper bound.

data['bus'].y → shape [num_buses, 2]:
- [0] va: Voltage angle solution (in radians) after power flow is solved; relative phase of voltage.
- [1] vm: Voltage magnitude solution (in per-unit) after power flow is solved.

---

data['generator'].x → shape [num_generators, 11]:
- [0] mbase: Generator’s power base rating (in MVA); used for scaling internal parameters.
- [1] pg: Scheduled or forecasted real power output (in MW).
- [2] pmin: Minimum real power the generator is allowed to produce.
- [3] pmax: Maximum real power limit.
- [4] qg: Scheduled reactive power output (in MVAr).
- [5] qmin: Minimum reactive power limit.
- [6] qmax: Maximum reactive power limit.
- [7] vg: Voltage magnitude setpoint for the bus this generator is controlling (applies if PV or ref bus).
- [8] c2: Coefficient of quadratic term in generator cost function (for pg²).
- [9] c1: Linear cost coefficient (for pg).
- [10] c0: Constant offset in cost function.

data['generator'].y → shape [num_generators, 2]:
- [0] pg: Real power output from the generator after solving the OPF (solution value).
- [1] qg: Reactive power output from the generator after solving OPF.

---

data['load'].x → shape [num_loads, 2]:
- [0] pd: Active power demand at this load (in MW).
- [1] qd: Reactive power demand (in MVAr); affects voltage and power factor.

---

data['shunt'].x → shape [num_shunts, 2]:
- [0] bs: Susceptance (imaginary admittance); controls how much reactive power is injected or absorbed.
- [1] gs: Conductance (real part of admittance); models energy loss at the shunt (real power dissipation).

---

## EDGE TYPES (All edges are typed and directional in HeteroData)

In this dataset, power flow and connectivity between nodes is modeled using edges. Each edge type includes:
- `edge_index`: [2, N] → source and target node indices
- `edge_attr`: [N, F] → physical properties of the connection (F = number of features)
- `edge_label`: [N, L] → solution values like power flows (L = number of labels)

⚠️ Do not use `.y` for edges — use `edge_label` for solution quantities like power flow.

---

### AC Line: `('bus', 'ac_line', 'bus')`  
Represents physical transmission lines between buses.

- `edge_index`: shape [2, num_ac_lines]
  - Row 0: source bus index (`from_bus`)
  - Row 1: target bus index (`to_bus`)
  - These indicate which two buses are connected by a line

- `edge_attr`: shape [num_ac_lines, 9]
  - [0] θ_l: Minimum allowed angle difference between buses
  - [1] θ_u: Maximum allowed angle difference
  - [2] b_from: Shunt susceptance on source bus side
  - [3] b_to: Shunt susceptance on target bus side
  - [4] br_r: Line resistance
  - [5] br_x: Line reactance
  - [6] rate_a: Normal thermal limit (MVA)
  - [7] rate_b: Emergency thermal limit
  - [8] rate_c: Absolute maximum thermal limit

- `edge_label`: shape [num_ac_lines, 4]
  - [0] pt: Real power flowing **toward the target bus**
  - [1] qt: Reactive power flowing **toward the target bus**
  - [2] pf: Real power flowing **from the source bus**
  - [3] qf: Reactive power flowing **from the source bus**

---

### Transformer: `('bus', 'transformer', 'bus')`  
Models two-winding transformers between buses with optional tap changers.

- `edge_index`: shape [2, num_transformers]
  - Row 0: source bus index
  - Row 1: target bus index

- `edge_attr`: shape [num_transformers, 11]
  - [0] θ_l: Min angle difference
  - [1] θ_u: Max angle difference
  - [2] br_r: Transformer resistance
  - [3] br_x: Transformer reactance
  - [4] rate_a: Normal rating
  - [5] rate_b: Emergency rating
  - [6] rate_c: Absolute max rating
  - [7] tap: Voltage magnitude tap ratio
  - [8] shift: Phase shift angle
  - [9] b_from: Charging susceptance from source bus
  - [10] b_to: Charging susceptance to target bus

- `edge_label`: shape [num_transformers, 4]
  - Same structure as AC lines: [pt, qt, pf, qf]

---

### Device-to-Bus Links (connectivity edges, no attributes or labels)

Used to connect component nodes (like generators or loads) to their parent buses. These edges define topology but do not carry physical parameters.

Each edge has:
- `edge_index`: [2, N]
  - Row 0: source index (component)
  - Row 1: target index (bus)

List of device edges:
- `('generator', 'generator_link', 'bus')`
- `('bus', 'generator_link', 'generator')`
- `('load', 'load_link', 'bus')`
- `('bus', 'load_link', 'load')`
- `('shunt', 'shunt_link', 'bus')`
- `('bus', 'shunt_link', 'shunt')`

These edges do not have:
- `edge_attr`
- `edge_label`

---

📌 Note:
- Use `edge_label` when querying power flow (not `.y`)
- Use `edge_index` to identify which nodes (e.g., buses) are involved in high-loading conditions
- Use link edges to map generators to buses (`data['generator', 'generator_link', 'bus'].edge_index`)

</user>
<broken-code>
{code_block}
</broken-code>
<error-message>
{error_message}
</error-message>
<code>
"""
