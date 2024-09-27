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
    alpha_S, a_S, theta_B, d_B = sy.symbols("alpha_S, a_S, theta_B, d_B", real=True)
    alpha_B, a_B, theta_0, d_0 = sy.symbols("alpha_B, a_B, theta_0, d_0", real=True)

    #
    # Station Frame -> Base Frame
    #
    print("*" * 20, "Forward Kinematics", "*" * 20)
    T_SB = dh_transformation(alpha_S, a_S, theta_B, d_B)
    print(f"T_SB = {T_SB}")
    T_SB = (
        T_SB.subs([(alpha_S, 0)]).subs([(a_S, 0)]).subs([(theta_B, 0)]).subs([(d_B, 1)])
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
    T_S0 = sy.simplify(T_SB * T_B0)
    print(f"T_S0 = {T_S0}")
    print()
    return T_S0


def main():
    T_S0 = forward_kinematics()


if __name__ == "__main__":
    main()
