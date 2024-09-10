import sympy as sy


def dh_transformation(alpha, a, theta, d):
    T_alpha = sy.Matrix(
        [
            [1, 0, 0, 0],
            [0, sy.cos(alpha), -sy.sin(alpha), 0],
            [0, sy.sin(alpha), sy.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )
    T_a = sy.Matrix(
        [
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T_theta = sy.Matrix(
        [
            [sy.cos(theta), -sy.sin(theta), 0, 0],
            [sy.sin(theta), sy.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T_d = sy.Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1],
        ]
    )
    return sy.simplify(T_alpha * T_a * T_theta * T_d)


def forward_kinematics():
    alpha_W, a_W, theta_B, d_B = sy.symbols("alpha_W, a_W, theta_B, d_B", real=True)
    alpha_B, a_B, theta_0, d_0 = sy.symbols("alpha_B, a_B, theta_0, d_0", real=True)

    #
    # Station Frame -> Base Frame
    #
    print("*" * 20, "Forward Kinematics", "*" * 20)
    T_SB = dh_transformation(alpha_W, a_W, theta_B, d_B)
    print(f"T_SB = {T_SB}")
    T_SB = (
        T_SB.subs([(alpha_W, 0)]).subs([(a_W, 0)]).subs([(theta_B, 0)]).subs([(d_B, 1)])
    )
    print(f"T_SB = {T_SB}")
    print()

    #
    # Base Frame -> Joint_0 Frame
    #
    T_B0 = dh_transformation(alpha_B, a_B, theta_0, d_0)
    print(f"T_B0 = {T_B0}")
    T_B0 = T_B0.subs([(alpha_B, -sy.pi / 2)]).subs([(a_B, 0)]).subs([(d_0, 0.101)])
    print(f"T_B0 = {T_B0}")
    print()

    #
    # Station Frame -> Joint_0 Frame
    #
    T_S0 = T_SB * T_B0
    print(f"T_S0 = {T_S0}")
    print()
    return T_S0


def eom(T_S0):
    theta_0 = sy.symbols("theta_0", real=True)
    theta_0_d = sy.symbols("theta_0_d", real=True)
    theta_0_dd = sy.symbols("theta_0_dd", real=True)

    l1 = sy.symbols("l1", real=True)
    m1 = sy.symbols("m1", real=True)
    I1 = sy.symbols("I1", real=True)
    g = sy.symbols("g", real=True)

    q = sy.Matrix([theta_0])
    q_d = sy.Matrix([theta_0_d])
    q_dd = sy.Matrix([theta_0_dd])

    #
    # Jacobian
    #
    print("*" * 20, "Jacobian", "*" * 20)
    C1 = T_S0 * sy.Matrix([l1, 0, 0, 1])
    C1 = sy.Matrix([C1[0], C1[1], C1[2]])
    J1 = C1.jacobian(q)

    print(f"C1 = {C1}")
    print(f"J1 = {J1}")
    print()

    #
    # Lagrangian
    #
    print("*" * 20, "Lagrangian", "*" * 20)
    C1_d = J1 * q_d
    T = sy.simplify(0.5 * m1 * C1_d.dot(C1_d) + 0.5 * I1 * theta_0_d**2)
    V = sy.simplify(m1 * g * C1[1])
    L = sy.simplify(T - V)

    print(f"C1_d = {C1_d}")
    print(f"T = {T}")
    print(f"V = {V}")
    print(f"L = {L}")

    #
    # Euler lagrange
    #
    dL_dq_d = []
    dt_dL_dq_d = []
    dL_dq = []
    EOM = []

    for i in range(len(q)):
        dL_dq_d.append(sy.diff(L, q_d[i]))

        temp = 0
        for j in range(len(q)):
            temp += (
                sy.diff(dL_dq_d[i], q[j]) * q_d[j]
                + sy.diff(dL_dq_d[i], q_d[j]) * q_dd[j]
            )
        dt_dL_dq_d.append(temp)
        dL_dq.append(sy.diff(L, q[i]))
        EOM.append(dt_dL_dq_d[i] - dL_dq[i])
    EOM = sy.simplify(sy.Matrix(EOM))

    print("*" * 20, "Euler lagrange", "*" * 20)
    for i in range(len(EOM)):
        print(f"EOM[{i}] = {EOM[i]}")
    print()

    #
    # 5. EOM M, C, G format
    #
    print("*" * 20, "M, C, G format", "*" * 20)
    M = EOM.jacobian(q_dd)
    b = EOM.subs([(theta_0_dd, 0)])
    G = b.subs([(theta_0_d, 0)])
    C = b - G

    for i in range(len(M)):
        print(f"M[{i + 1}] = {M[i]}")
    for i in range(len(C)):
        print(f"C[{i + 1}] = {C[i]}")
    for i in range(len(G)):
        print(f"G[{i + 1}] = {G[i]}")
    print()


def main():
    T_S0 = forward_kinematics()
    eom(T_S0)


if __name__ == "__main__":
    main()
