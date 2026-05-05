
import numpy as np

def test(name, lst):
    print(f"--- Testing {name} ---")
    try:
        arr = np.array([lst])
        print(f"Success. Shape: {arr.shape}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Scenario 1: Mixed float and numpy float scalar
r_mixed = [[np.float64(1.0)]] * 159 + [[0.0]]
test("Start float64, End float", r_mixed)

# Scenario 2: Mixed bool and float
t_mixed = [[True]] * 159 + [[1.0]]
test("Start bool, End float", t_mixed)

# Scenario 3: Mixed array(scalar) and float
r_array = [[np.array(1.0)]] * 159 + [[0.0]]
test("Start array(scalar), End float", r_array)

# Scenario 4: Mixed array(1D) and float
r_array_1d = [[np.array([1.0])]] * 159 + [[0.0]]
test("Start array(1D), End float", r_array_1d)

# Scenario 5: Mixed list of array and list of float
# r.append([reward]) -> reward is array(1.0). [array(1.0)]
# r.append([0.]) -> [0.]
r_list_diff = [[np.array(1.0)]] * 159 + [[0.0]]
test("List of array(scalar) vs List of float", r_list_diff)

# Scenario 6: nested
# r.append([ [1.0] ]) vs r.append([ 0.0 ])
nested = [ [[1.0]] ] * 159 + [ [0.0] ]
test("Nested vs Flat", nested)
