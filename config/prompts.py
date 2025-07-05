
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

## EDGE TYPES

### AC Line: ('bus', 'ac_line', 'bus')  
Models physical transmission lines between buses.

edge_index → shape [2, num_ac_lines]:
- Row 0: Source bus index (from bus)
- Row 1: Destination bus index (to bus)

edge_attr → shape [num_ac_lines, 9]:
- [0] θ_l: Minimum allowed voltage angle difference between source and target bus.
- [1] θ_u: Maximum allowed voltage angle difference.
- [2] b_from: Shunt charging susceptance on the "from" bus side.
- [3] b_to: Shunt charging susceptance on the "to" bus side.
- [4] br_r: Series resistance of the line (causes real power loss).
- [5] br_x: Series reactance of the line (affects voltage drop and reactive flow).
- [6] rate_a: Maximum continuous thermal limit (in MVA).
- [7] rate_b: Thermal limit under contingency (emergency).
- [8] rate_c: Absolute maximum limit under extreme conditions.

edge_label → shape [num_ac_lines, 4]:
- [0] pt: Real power flowing **toward** the destination bus (MW).
- [1] qt: Reactive power flowing **toward** the destination bus (MVAr).
- [2] pf: Real power flowing **from** the source bus.
- [3] qf: Reactive power flowing **from** the source bus.

---

### Transformer: ('bus', 'transformer', 'bus')  
Models a two-winding transformer with tap control and angle shift.

edge_index → shape [2, num_transformers]:
- Row 0: Source bus index
- Row 1: Destination bus index

edge_attr → shape [num_transformers, 11]:
- [0] θ_l: Min angle difference allowed across transformer.
- [1] θ_u: Max angle difference.
- [2] br_r: Transformer resistance (series loss).
- [3] br_x: Transformer reactance (impedance).
- [4] rate_a: Continuous thermal rating (in MVA).
- [5] rate_b: Emergency thermal rating.
- [6] rate_c: Absolute thermal rating.
- [7] tap: Tap ratio (voltage magnitude scaling).
- [8] shift: Phase shift (in degrees or radians).
- [9] b_from: Charging susceptance at the source bus side.
- [10] b_to: Charging susceptance at the target bus side.

edge_label → shape [num_transformers, 4]:
- [0] pt: Real power flowing **toward** the destination bus (MW).
- [1] qt: Reactive power flowing **toward** the destination bus (MVAr).
- [2] pf: Real power flowing **from** the source bus.
- [3] qf: Reactive power flowing **from** the source bus

---

### Component Links (no features or labels)
These edges are used only to attach devices to their corresponding bus. They are directional connections.

edge_index → shape [2, N]:
- Row 0: Source node index (generator/load/shunt)
- Row 1: Target node index (bus)

Available link types:
- ('generator', 'generator_link', 'bus')
- ('bus', 'generator_link', 'generator')
- ('load', 'load_link', 'bus')
- ('bus', 'load_link', 'load')
- ('shunt', 'shunt_link', 'bus')
- ('bus', 'shunt_link', 'shunt')

# CODING RULES:
- Do NOT assume any labels yourself in the data.
- Don't give functions in the code. 
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

## EDGE TYPES

### AC Line: ('bus', 'ac_line', 'bus')  
Models physical transmission lines between buses.

edge_index → shape [2, num_ac_lines]:
- Row 0: Source bus index (from bus)
- Row 1: Destination bus index (to bus)

edge_attr → shape [num_ac_lines, 9]:
- [0] θ_l: Minimum allowed voltage angle difference between source and target bus.
- [1] θ_u: Maximum allowed voltage angle difference.
- [2] b_from: Shunt charging susceptance on the "from" bus side.
- [3] b_to: Shunt charging susceptance on the "to" bus side.
- [4] br_r: Series resistance of the line (causes real power loss).
- [5] br_x: Series reactance of the line (affects voltage drop and reactive flow).
- [6] rate_a: Maximum continuous thermal limit (in MVA).
- [7] rate_b: Thermal limit under contingency (emergency).
- [8] rate_c: Absolute maximum limit under extreme conditions.

edge_label → shape [num_ac_lines, 4]:
- [0] pt: Real power flowing **toward** the destination bus (MW).
- [1] qt: Reactive power flowing **toward** the destination bus (MVAr).
- [2] pf: Real power flowing **from** the source bus.
- [3] qf: Reactive power flowing **from** the source bus.

---

### Transformer: ('bus', 'transformer', 'bus')  
Models a two-winding transformer with tap control and angle shift.

edge_index → shape [2, num_transformers]:
- Row 0: Source bus index
- Row 1: Destination bus index

edge_attr → shape [num_transformers, 11]:
- [0] θ_l: Min angle difference allowed across transformer.
- [1] θ_u: Max angle difference.
- [2] br_r: Transformer resistance (series loss).
- [3] br_x: Transformer reactance (impedance).
- [4] rate_a: Continuous thermal rating (in MVA).
- [5] rate_b: Emergency thermal rating.
- [6] rate_c: Absolute thermal rating.
- [7] tap: Tap ratio (voltage magnitude scaling).
- [8] shift: Phase shift (in degrees or radians).
- [9] b_from: Charging susceptance at the source bus side.
- [10] b_to: Charging susceptance at the target bus side.

edge_label → shape [num_transformers, 4]:
- [0] pt: Real power flowing **toward** the destination bus (MW).
- [1] qt: Reactive power flowing **toward** the destination bus (MVAr).
- [2] pf: Real power flowing **from** the source bus.
- [3] qf: Reactive power flowing **from** the source bus.

---

### Component Links (no features or labels)
These edges are used only to attach devices to their corresponding bus. They are directional connections.

edge_index → shape [2, N]:
- Row 0: Source node index (generator/load/shunt)
- Row 1: Target node index (bus)

Available link types:
- ('generator', 'generator_link', 'bus')
- ('bus', 'generator_link', 'generator')
- ('load', 'load_link', 'bus')
- ('bus', 'load_link', 'load')
- ('shunt', 'shunt_link', 'bus')
- ('bus', 'shunt_link', 'shunt')
</user>
<broken-code>
{code_block}
</broken-code>
<error-message>
{error_message}
</error-message>
<code>
"""
