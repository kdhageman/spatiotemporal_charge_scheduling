import pyomo.environ as pyo

from util.parameters import get_T_W, get_T_S


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

    T_W = get_T_W(N_d, N_w, positions_w)
    T_S = get_T_S(N_d, N_s, N_w, positions_S, positions_w)

    # MODEL DEFINITION
    model = pyo.ConcreteModel()

    model.d = pyo.RangeSet(0, 0)
    model.s = pyo.RangeSet(0, N_s - 1)
    model.w = pyo.RangeSet(0, N_w - 1)
    model.w_s = pyo.RangeSet(0, N_w - 2)
    model.w_d = pyo.RangeSet(1, N_w - 1)
    model.T_W = T_W
    model.T_S = T_S

    # VARIABLES
    model.x = pyo.Var(model.d, model.s, model.w_s, domain=pyo.Binary)
    model.D = pyo.Var(model.d, model.s, model.w_s, domain=pyo.NonNegativeReals)
    model.b_arr = pyo.Var(model.d, model.w)
    model.b_min = pyo.Var(model.d, model.s, model.w_s)
    model.b_plus = pyo.Var(model.d, model.s, model.w_s)

    # CONSTRAINTS
    for d in model.d:
        model.b_arr[d, 0] = B_start[d]
        model.b_arr[d, 0].fix()

    def rule_at_most_one_station_per_drone(model, d, w_s):
        return sum(model.x[d, s, w_s] for s in model.s) <= 1

    model.at_most_one_station_per_drone = pyo.Constraint(model.d, model.w_s, rule=rule_at_most_one_station_per_drone)

    def rule_b_min_calc(model, d, s, w_s):
        return model.b_min[d, s, w_s] == model.b_arr[d, w_s] - T_S[d, s, w_s] * r_deplete[d]

    model.b_min_calc = pyo.Constraint(model.d, model.s, model.w_s, rule=rule_b_min_calc)

    def rule_b_plus_calc(model, d, s, w_s):
        return model.b_plus[d, s, w_s] == model.b_min[d, s, w_s] + model.D[d, s, w_s] * r_charge[d]

    model.b_plus_calc = pyo.Constraint(model.d, model.s, model.w_s, rule=rule_b_plus_calc)

    def rule_b_arr_calc(model, d, w_d):
        # battery when coming directly from previous waypoint
        def direct(d, w_d):
            return (1 - sum(model.x[d, s, w_d - 1] for s in model.s)) * (
                    model.b_arr[d, w_d - 1] - T_W[d, w_d - 1] * r_deplete[d])

        # battery when going via a charging station
        def via_s(d, w_d):
            return sum(
                model.x[d, s, w_d - 1] * (model.b_plus[d, s, w_d - 1] - T_S[d, s, w_d] * r_deplete[d]) for s in model.s)

        return model.b_arr[d, w_d] == direct(d, w_d) + via_s(d, w_d)

    model.b_arr_calc = pyo.Constraint(model.d, model.w_d, rule=rule_b_arr_calc)

    # upper and lower limits of each values
    def rule_D_ulim(model, d, s, w_s):
        return model.D[d, s, w_s] <= D_max[d]

    model.D_ulim = pyo.Constraint(model.d, model.s, model.w_s, rule=rule_D_ulim)

    def rule_b_arr_llim(model, d, w):
        return model.b_arr[d, w] >= B_min

    model.b_arr_llim = pyo.Constraint(model.d, model.w, rule=rule_b_arr_llim)

    def rule_b_min_llim(model, d, s, w_s):
        return model.b_min[d, s, w_s] * model.x[d, s, w_s] >= B_min

    model.b_min_llim = pyo.Constraint(model.d, model.s, model.w_s, rule=rule_b_min_llim)

    def rule_b_plus_ulim(model, d, s, w_s):
        return model.b_plus[d, s, w_s] * model.x[d, s, w_s] <= B_max

    model.b_plus_ulim = pyo.Constraint(model.d, model.s, model.w_s, rule=rule_b_plus_ulim)

    # OBJECTIVE
    def P(d):
        def direct(w_s):
            return (1 - sum(model.x[d, s, w_s] for s in model.s)) * T_W[d, w_s]

        def via(s, w_s):
            return model.x[d, s, w_s] * (T_S[d, s, w_s] + T_S[d, s, w_s + 1] + model.D[d, s, w_s])

        return sum(direct(w_s) + sum(via(s, w_s) for s in model.s) for w_s in model.w_s)

    model.execution_time = pyo.Objective(
        expr=sum(P(d) for d in model.d),
        sense=pyo.minimize,
    )

    return model
