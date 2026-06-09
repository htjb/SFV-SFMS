"""Prior for analysis of SHO and Ext-Reg PSD models."""

prior = {
    "s0": {"min": 0.0001, "max": 5},
    "tau0": {"min": 0.001, "max": 3.0},
    "Q": {"min": 0.51, "max": 5.0},
    "sig_reg": {"min": 0.01, "max": 10.0},
    "tau_eq": {"min": 0.001, "max": 10.0},
    "tau_in": {"min": 0.01, "max": 1.0},
    "sig_dyn": {"min": 0.01, "max": 5.0},
    "tau_dyn": {"min": 0.001, "max": 0.1},
    "noise": {"min": 0.001, "max": 0.35},
}
