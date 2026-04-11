# LCS-PD-SMC: Integrating Derivative Control and Sliding Mode Control into Latent Centroid Steering

## 1. Background

### 1.1 Current System: LCS (Latent Centroid Steering)

LCS is a single-pass guidance mechanism that replaces instance-level CFG residuals with pre-computed class-level centroid shifts. Applied in the latent space of the language model (`driving_features`):

$$
z' = z + \gamma \cdot \delta_c, \quad \delta_c = \mu_c - \mu_{\text{global}}
$$

where:
- $z$: conditional driving features from the LLM (shape `[B, 30, H]`)
- $\gamma$: scale factor (`lmsi_scale`, currently 0.10)
- $\delta_c = \mu_c - \mu_{\text{global}}$: pre-computed centroid shift for command class $c \in \{1,...,6\}$

**Implementation**: `DrivingModel._apply_lcs()` in `simlingo_training/models/driving.py`

**Properties**:
- Single-pass (no unconditional forward needed)
- Static per command class (same $\delta_c$ regardless of driving state)
- Equivalent to **P-control** with constant gain $\gamma$ in the CFG-Ctrl framework

### 1.2 CFG-Ctrl Framework (arXiv:2603.03281)

CFG-Ctrl reinterprets Classifier-Free Guidance as feedback control:

$$
e = v_{\text{cond}} - v_{\text{uncond}} \quad \text{(error signal)}
$$

$$
v_{\text{guided}} = v_{\text{uncond}} + K(x,t) \cdot e \quad \text{(state-dependent gain)}
$$

Key insight: standard CFG is a **proportional controller** with fixed gain. The paper proposes **SMC-CFG** (Sliding Mode Control) which introduces:
- A sliding surface: $s(t) = \dot{e}(t) + \lambda \cdot e(t)$
- A switching control term: $\Delta e = -k \cdot \text{sign}(s(t))$
- Lyapunov-stable finite-time convergence guarantees

### 1.3 Motivation for Integration

LCS suffers from two limitations:
1. **No temporal dynamics**: The shift $\delta_c$ is identical whether the car just entered an intersection or is mid-turn. Command transitions are abrupt.
2. **No safety bounds**: High $\gamma$ can push the latent arbitrarily far, causing erratic waypoints.

The CFG-Ctrl paper's control-theoretic perspective provides principled solutions: derivative control for dynamic adaptability, and SMC for safety-aware boundaries.

---

## 2. Proposed Formulation: LCS-PD-SMC

### 2.1 Target Formula

Combining proportional steering, derivative stabilization, and safety constraints:

$$
z' = z + \alpha \delta_t + \beta(\delta_t - \delta_{t-1}) - k \tanh(S(z))
$$

where:
- $z$: conditional driving features (LLM hidden state for driving adaptor tokens)
- $\delta_t$: centroid shift at agent step $t$
- $\alpha$: proportional gain (replaces $\gamma$)
- $\beta$: derivative gain
- $k$: SMC switching gain
- $S(z)$: sliding mode surface. In discrete agent steps we use $S_t$ (**Design 1 — error dynamics**, §3.3). Let $e_t = \alpha\delta_t + \beta(\delta_t-\delta_{t-1})$ (PD correction before SMC). Then
  $$
  S_t = (e_t - e_{t-1}) + \lambda\, e_t
  $$
  The $z$ in $S(z)$ is shorthand for “at the current latent steering step”; it is the same $S_t$ as in `driving_pdsmc.py`. Alternative surfaces (norm / action-space) are listed in §3.3.
- $\tanh$: smooth approximation of $\text{sign}$ (avoids chattering in discrete 20 FPS system)

### 2.2 Component Analysis

| Term | Role | CFG-Ctrl Analogy | Effect in Driving |
|------|------|------------------|-------------------|
| $\alpha \delta_t$ | Proportional steering | P-control (existing LCS) | Shifts latent toward command centroid |
| $\beta(\delta_t - \delta_{t-1})$ | Derivative stabilization | D-control (new) | Smooth command transitions, anticipatory steering |
| $-k \tanh(S(z))$ | SMC safety boundary | Sliding Mode Control (new) | Prevents overshooting, enforces safe corridor |

---

## 3. Component Design

### 3.1 Defining $\delta_t$ Across Agent Steps

**Option A — Command-gated delta** (recommended for initial implementation):
- $\delta_t = \delta_{c_t}$ where $c_t$ is the navigation command at step $t$
- $\delta_t - \delta_{t-1}$ is **non-zero only when the command changes** (e.g., cmd 4 "follow road" → cmd 1 "turn left")
- Makes the D-term a **command-transition smoother**
- Preserves single-pass property

**Option B — State-modulated delta** (richer, for future exploration):
- $\delta_t = f(v_t, d_t) \cdot \delta_{c_t}$ where $f$ depends on speed $v_t$ and distance to next waypoint $d_t$
- Even with the same command, $\delta_t$ varies across steps
- Provides true temporal derivative that dampens oscillation

The existing agent code already tracks command transitions:
```python
# In agent tick():
if self.last_command_tmp != far_command:
    self.last_command = self.last_command_tmp
self.last_command_tmp = far_command
self.current_far_command = far_command
```

This `last_command` → `far_command` transition provides the $c_{t-1}$ → $c_t$ information needed for the D-term.

### 3.2 Derivative Term: $\beta(\delta_t - \delta_{t-1})$

**Physical meaning**: When command changes from "follow road" ($c=4$) to "turn left" ($c=1$), the difference $\delta_1 - \delta_4$ is a vector in latent space representing the *direction of command transition*.

**Behavior**:
- **At command transitions** ($c_t \neq c_{t-1}$): Provides anticipatory boost ($\beta > 0$) to help the model respond faster to new commands. Directly addresses the "command-following gap."
- **During steady-state** ($c_t = c_{t-1}$): $\delta_t - \delta_{t-1} = 0$, D-term vanishes. System degrades gracefully to LCS + SMC safety.
- **Rapid command sequences** (intersection → straight → intersection): D-term provides damping that prevents the latent from whipsawing.

**Note on persistence**: The derivative term fires for exactly one step at each command transition (when using Option A). To sustain the effect over multiple steps, consider exponential decay: $\beta_t = \beta_0 \cdot \exp(-\eta \cdot (t - t_{\text{transition}}))$.

### 3.3 SMC Safety Term: $-k \tanh(S(z))$

#### Sliding Surface Definitions

**Design 1 — Error-dynamics surface** (closest to SMC-CFG paper, recommended):

Define the "error" as the total PD correction: $e_t = \alpha \delta_t + \beta(\delta_t - \delta_{t-1})$

$$
S_t = (e_t - e_{t-1}) + \lambda \cdot e_t
$$

This encodes the target dynamics: the correction should converge (decay) over time. The SMC term $-k \tanh(S_t)$ enforces convergence toward this sliding manifold:
- If correction is **growing AND already large** → $S$ is large positive → $\tanh(S) \approx 1$ → pushes back strongly
- If correction is **shrinking AND small** → $S$ near zero → minimal intervention

**Design 2 — Latent-norm surface** (simpler, interpretable):

$$
S(z) = \frac{\| z' - z \|}{\| z \|} - \tau
$$

where $\tau$ is a safety threshold (e.g., 0.1 = max 10% relative shift). This directly constrains the relative magnitude of the total LCS correction.

**Design 3 — Action-space safety surface** (post-hoc, can layer on top):

After decoding waypoints $wp'$ from steered latent:

$$
S = \| wp' - wp_{\text{baseline}} \| - d_{\max}
$$

where $wp_{\text{baseline}}$ is the unsteered prediction and $d_{\max}$ is maximum allowed deviation (e.g., 2 meters). Directly prevents LCS from pushing waypoints beyond a safe corridor.

**Recommendation**: Use Design 1 for the core latent-space mechanism. Optionally layer Design 3 as a hard safety clamp in action space.

#### Why $\tanh$ instead of $\text{sign}$

The original SMC-CFG uses $\text{sign}(s)$ which creates discontinuous switching. In the discrete 20 FPS driving system, this causes "chattering" — rapid oscillation of the correction. $\tanh$ is a standard boundary-layer approximation from SMC literature that smooths the switching function while preserving the convergence property.

---

## 4. Implementation Plan

### 4.1 New Config Parameters

In `team_code/config_simlingo_command_meanshift_conditional.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lcs_alpha` | 0.10 | Proportional gain (replaces `mean_shift_scale`) |
| `lcs_beta` | 0.05 | Derivative gain (command transition boost) |
| `lcs_k` | 0.02 | SMC switching gain |
| `lcs_lambda` | 1.0 | Sliding surface shape parameter |
| `lcs_use_derivative` | `False` | Enable derivative term |
| `lcs_use_smc` | `False` | Enable SMC safety term |

### 4.2 Agent-Level State Tracking

In `agent_simlingo_cluster_command_mean_shift_centroid_conditional.py`, add state variables in `setup()`:

```python
# LCS-PD-SMC temporal state
self.prev_cmd_id = None          # c_{t-1}
self.prev_delta = None           # δ_{t-1} (tensor, shape [30, H])
self.prev_correction = None      # e_{t-1} (total PD correction from previous step)
```

In `run_step()`, after model forward, update state:

```python
# Store temporal state for next step
self.prev_cmd_id = current_cmd_id
self.prev_delta = current_delta       # the δ_{c_t} used this step
self.prev_correction = current_corr   # the α*δ_t + β*Δδ applied this step
```

### 4.3 Enhanced LCS Method

In `simlingo_training/models/driving.py`, replace or extend `_apply_lcs`:

```python
def _apply_lcs_pdsmc(self, driving_features, prompt_str,
                      prev_delta=None, prev_correction=None,
                      alpha=0.10, beta=0.05, k=0.02, lambda_smc=1.0,
                      use_derivative=True, use_smc=True):
    """
    LCS with PD control + SMC safety:
    z' = z + α*δ_t + β*(δ_t - δ_{t-1}) - k*tanh(S)
    """
    # 1. Look up centroid shift δ_t
    cmd_id = self._parse_command_id(prompt_str)
    if cmd_id is None:
        cmd_id = getattr(self, '_lmsi_command_override', None)
    if cmd_id is None or cmd_id not in self._lmsi_priors:
        return driving_features, None, None

    delta_t = self._lmsi_priors[cmd_id].to(
        device=driving_features.device, dtype=driving_features.dtype
    )  # [30, H]

    # 2. P-term: α * δ_t
    p_term = alpha * delta_t.unsqueeze(0)

    # 3. D-term: β * (δ_t - δ_{t-1})
    if use_derivative and prev_delta is not None:
        d_term = beta * (delta_t - prev_delta).unsqueeze(0)
    else:
        d_term = torch.zeros_like(p_term)

    # 4. Total PD correction (before SMC)
    correction = p_term + d_term  # [1, 30, H]

    # 5. SMC safety term: -k * tanh(S)
    if use_smc and prev_correction is not None:
        # Sliding surface: S = (correction - prev_correction) + λ * correction
        S = (correction - prev_correction) + lambda_smc * correction
        smc_term = -k * torch.tanh(S)
    else:
        smc_term = torch.zeros_like(correction)

    # 6. Final steered latent
    z_steered = driving_features + correction + smc_term

    return z_steered, delta_t, correction + smc_term
```

### 4.4 Agent-Model Interface

In `run_step()`, pass temporal state to the model before forward:

```python
# Before model forward
self.model._lcs_prev_delta = self.prev_delta
self.model._lcs_prev_correction = self.prev_correction
self.model._lcs_params = {
    'alpha': self.config.lcs_alpha,
    'beta': self.config.lcs_beta,
    'k': self.config.lcs_k,
    'lambda_smc': self.config.lcs_lambda,
    'use_derivative': self.config.lcs_use_derivative,
    'use_smc': self.config.lcs_use_smc,
}

# Model forward
pred_speed_wps, pred_route, language = self.model(model_input)

# After model forward: retrieve and store state for next step
self.prev_delta = getattr(self.model, '_lcs_current_delta', None)
self.prev_correction = getattr(self.model, '_lcs_current_correction', None)
```

### 4.5 Control Flow Per Step

```
Step t of run_step():
  1. tick() → compute command c_t, build prompt, get model inputs
  2. Set model temporal state (prev_delta, prev_correction)
  3. model.forward() → inside _apply_lcs_pdsmc():
     a. Look up δ_t = δ_{c_t} from pre-computed priors
     b. P-term: α * δ_t
     c. D-term: β * (δ_t - δ_{t-1})     [zero if same command or first step]
     d. PD correction: corr = P + D
     e. Sliding surface: S = (corr - corr_{t-1}) + λ * corr
     f. SMC term: -k * tanh(S)
     g. Final: z' = z + corr + SMC
     h. Store (δ_t, corr+SMC) as current state
  4. Decode z' → waypoints → PID → vehicle control
  5. Update agent state: prev_delta ← δ_t, prev_correction ← corr+SMC
```

---

## 5. Design Decisions and Trade-offs

### 5.1 Single-Pass Property Preserved

The PD+SMC extension remains single-pass because it only uses:
- Pre-computed centroids (same as LCS)
- Temporal state from previous agent step (stored in agent, no extra forward)

No unconditional forward pass is needed.

### 5.2 Interaction with Existing PID Controller

The system has **two layers** of control:
- **Latent space** (LCS-PD-SMC): Shapes *what trajectory* to follow
- **Action space** (speed PID + lateral PID): Shapes *how to follow* the trajectory

These don't conflict — latent-space control operates upstream of the waypoint decoder, while action-space PID operates downstream on the decoded waypoints.

### 5.3 D-Term Persistence

With command-gated deltas (Option A), the D-term fires for exactly one step at each command transition. For multi-step transitions, consider exponential decay:

$$
\beta_t = \beta_0 \cdot \exp\left(-\eta \cdot (t - t_{\text{transition}})\right)
$$

This spreads the derivative boost over multiple steps, providing smoother transitions.

### 5.4 SMC Gain $k$ Sensitivity

From the CFG-Ctrl paper's ablation (Table 3):
- Too small $k$: slow convergence, weaker text-image alignment (in our case: weaker command following)
- Too large $k$: abrupt corrections, reduced quality

For driving, start conservative ($k = 0.02$) and increase. The `tanh` smoothing mitigates chattering but $k$ still controls the maximum correction magnitude.

### 5.5 When to Disable Components

| Scenario | Configuration |
|----------|--------------|
| Baseline comparison | `use_derivative=False, use_smc=False` (pure LCS) |
| Smooth transitions only | `use_derivative=True, use_smc=False` |
| Safety bounds only | `use_derivative=False, use_smc=True` |
| Full system | `use_derivative=True, use_smc=True` |

---

## 6. Ablation Strategy

Progressive evaluation on CARLA benchmarks:

1. **Baseline**: Current LCS (P only, $\alpha=0.10$)
2. **+D**: Add derivative term ($\beta=0.05$) → measure command-following improvement at transitions (intersection success rate)
3. **+SMC**: Add SMC safety ($k=0.02, \lambda=1.0$) → measure robustness under high $\alpha$ values
4. **+PD+SMC**: Full Eq. 11 → measure combined benefit on driving score

### Key Metrics to Track
- **Route completion**: Does PD+SMC improve or degrade overall completion?
- **Intersection success rate**: Does the D-term help at decision points?
- **Command-following accuracy**: Does the system turn when told to turn?
- **Infraction rate**: Does SMC reduce erratic behavior from overshooting?
- **Latent shift magnitude**: Monitor $\|z' - z\|$ to verify SMC is bounding corrections

---

## 7. Theoretical Justification

### 7.1 Lyapunov Stability (from CFG-Ctrl)

With the sliding surface $S_t = (e_t - e_{t-1}) + \lambda e_t$ and Lyapunov function $V(S) = \frac{1}{2}\|S\|^2$:

$$
\dot{V} = S^\top \dot{S} \leq -\eta \|S\|, \quad \eta = k \cdot b_{\min} - \delta > 0
$$

This guarantees finite-time convergence: $\|S(t)\| = 0$ for some $t \leq \frac{\|S(0)\|}{\eta}$.

In the driving domain, this means the correction converges to a stable manifold where:
- The PD correction is decaying (error dynamics are healthy)
- The system is not overshooting or oscillating

### 7.2 Connection to CFG-Ctrl Table 1

Our LCS-PD-SMC maps to the CFG-Ctrl framework as:

| Component | $K_t$ (Gain) | $\Pi_t$ (Operator) | Error $e(t)$ |
|-----------|-------------|-------------------|-------------|
| LCS (P) | $\alpha$ | $I$ | $\delta_{c_t}$ |
| LCS-PD | $[\alpha I \;\; \beta I]$ | $I$ | $[\delta_t \;\; \delta_t - \delta_{t-1}]^\top$ |
| LCS-PD-SMC | $[\alpha I \;\; \beta I]$ | $I$ | $\delta_t - k \cdot \text{sign}(S_t)$ |

The key difference from standard SMC-CFG: our error signal $e(t)$ is a **class-level centroid** rather than an instance-level conditional-unconditional residual, preserving the single-pass computational advantage.
