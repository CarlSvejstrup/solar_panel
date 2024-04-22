from sympy import symbols, Piecewise

x, y = symbols('x y')

# Define a piecewise function of two variables
f = Piecewise(
    (x*y, x + y < 1),
    (x**2 + y**2, x + y >= 1)
)

# Integrate the piecewise function over the region where x and y are both between 0 and 1
result = Piecewise.piecewise_integrate(f, (x, 0, 1), (y, 0, 1))

print(result)