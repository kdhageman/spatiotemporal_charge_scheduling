import pyomo.environ as pyo

from util.second.parameters import get_T_N, get_T_W


def get_model(indices={}, parameters={}):
    # extract from function parameters
    N_d = indices["N_d"]
    N_s = indices["N_s"]
    N_w = indices["N_w"]

    B_start = parameters["B_start"]
    B_min = parameters["B_min"]
    B_max = parameters["B_max"]

    m = parameters["m"]  # TODO: use!
    r_charge = parameters['r_charge']
    r_deplete = parameters['r_deplete']

    D_max = (B_max - B_min) / r_charge

    positions_S = parameters['positions_S']
    positions_w = parameters['positions_w']

    T_N = get_T_N(N_d, N_s, N_w - 1, positions_S, positions_w)
    T_W = get_T_W(N_d, N_s, N_w - 1, positions_S, positions_w)

    # MODEL DEFINITION
    model = pyo.ConcreteModel()

    model.d = pyo.RangeSet(0, N_d - 1)
    model.s = pyo.RangeSet(0, N_s - 1)
    model.w = pyo.RangeSet(0, N_w - 1)
    model.w_s = pyo.RangeSet(0, N_w - 2)
    model.w_d = pyo.RangeSet(1, N_w - 1)
    model.n = pyo.RangeSet(0, N_s)
    model.T_N = T_N
    model.T_W = T_W

    # VARIABLES
    model.P = pyo.Var(model.d, model.n, model.w_s, domain=pyo.Binary)
    model.D = pyo.Var(model.d, model.w_s, domain=pyo.NonNegativeReals)
    model.b_arr = pyo.Var(model.d, model.w)
    model.b_min = pyo.Var(model.d, model.w_s)
    model.b_plus = pyo.Var(model.d, model.w_s)

    # CONSTRAINTS
    model.path_constraint = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: sum(m.P[d, n, w_s] for n in model.n) == 1
    )

    # battery constraints
    print(f"model.d: {[d for d in model.d]}")
    model.b_arr_start = pyo.Constraint(
        model.d,
        rule=lambda m, d: m.b_arr[d, 0] == B_start[d]
    )

    model.b_min_calc = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.b_min[d, w_s] == m.b_arr[d, w_s] - r_deplete[d] * sum(
            m.P[d, n, w_s] * T_N[d, n, w_s] for n in m.n)
    )

    model.b_plus_calc = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.b_plus[d, w_s] == m.b_min[d, w_s] + r_charge[d] * m.D[d, w_s]
    )

    model.b_arr_calc = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.b_arr[d, w_s + 1] == m.b_plus[d, w_s] - r_deplete[d] * sum(
            m.P[d, n, w_s] * T_W[d, n, w_s] for n in m.n)
    )

    # lower and upper bounds of variables values
    model.b_arr_llim = pyo.Constraint(
        model.d,
        model.w,
        rule=lambda m, d, w: m.b_arr[d, w] >= B_min
    )

    model.b_min_llim = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.b_min[d, w_s] >= B_min
    )
    model.b_plus_ulim = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.b_plus[d, w_s] <= B_max
    )

    model.D_lim = pyo.Constraint(
        model.d,
        model.w_s,
        rule=lambda m, d, w_s: m.D[d, w_s] <= (1 - m.P[d, N_s, w_s]) * D_max[d]
    )

    # OBJECTIVE
    def E(d):
        return sum(
            model.D[d, w_s] + sum(model.P[d, n, w_s] * (T_N[d, n, w_s] + T_W[d, n, w_s]) for n in model.n) for w_s in model.w_s)

    model.execution_time = pyo.Objective(
        expr=sum(E(d) for d in model.d),
        sense=pyo.minimize,
    )

    return model
