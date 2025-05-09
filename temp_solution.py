import sympy as sp

def complex_distance(z1, z2):
    # Extract real and imaginary parts
    x1, y1 = sp.re(z1), sp.im(z1)
    x2, y2 = sp.re(z2), sp.im(z2)
    
    # Calculate differences
    dx = x2 - x1
    dy = y2 - y1
    
    # Square the differences and sum them
    sum_of_squares = dx**2 + dy**2
    
    # Take the square root of the sum
    distance = sp.sqrt(sum_of_squares)
    
    return distance

# Joe's and Gracie's points
joe_point = 1 + 2*sp.I
gracie_point = -1 + sp.I

# Calculate the distance between the points
distance = complex_distance(joe_point, gracie_point)

def main():
    joe_point = 1 + 2*sp.I
    gracie_point = -1 + sp.I
    distance = complex_distance(joe_point, gracie_point)
    return distance

if __name__ == "__main__":
    print(main())