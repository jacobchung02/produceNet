import numpy as np

p = .2
v = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
A = np.array(
    [
        [0, .5, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, .5, .5, 0, 0, 0],
        [0, .5, 0, 0.5, 0, 0, 0, .5],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0.5, 0, 0, 0.5],
        [0, 0, 0, 0, 0, 0.5, 0, 0]
    ])

ones = np.ones((8, 8))
B = 1/8 * ones
M = ((1-p)*A) + (p*B)

print('this is intial rank_vector')
print(v, '\n')
print('this is matrix A')
print(A, '\n')
print('this is matrix B')
print(B, '\n')
print('this is matrix M')
print(M)

def multiple(A, v, n):
    result = v
    for _ in range(n):
        result = A @ result
    return result

for i in range (15):
  final_result = multiple(M, v, i)
  print(f"Result for i = {i}:\n{final_result}")
  print('\n')

