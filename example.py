import numpy as np
import sympy

C = 1
L = 1

X = np.random.rand(100, 2) * 4 - 2
y = C * np.sinh(np.pi / L * X[:, 0]) * np.exp(-np.pi / L * X[:, 1])

from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=40,
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "cos",
        "sin",
        "exp",
        # "inv(x) = 1/x",
        # "both(x) = 2 * cos(x) * sin(x)",
        # ^ Custom operator (julia syntax)
    ],
    # extra_sympy_mappings={"inv": lambda x: 1 / x, "both": lambda x: (2 * sympy.cos(x) * sympy.sin(x))},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    # bin_op_weight=[0.5, 0.5],
    # un_op_weight=[0, 0.15, 0.15, 0.7],
)

model.fit(X, y)

print(model)
print(model.loss(X, y))
