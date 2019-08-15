import numpy as np


def sigmoid(z_value):
    return 1.0/(1.0+np.exp(-z_value))


def z(w, a_value, b):
    return w * a_value + b


def a(z_value):
    return sigmoid(z_value)


def sigmoid_prime(z_value):
    return sigmoid(z_value)*(1-sigmoid(z_value))


def dc_db(z_value, dc_da):
    return sigmoid_prime(z_value) * dc_da


def dc_dw(a_prev, dc_db_value):
    return a_prev * dc_db_value


def dc_da_prev(w, dc_db_value):
    return w * dc_db_value


a_l0 = 0.8
w_l1 = 1.58
b_l1 = -0.14
print(f"w_l1 = {round(w_l1, 4)}")
print(f"b_l1 = {round(b_l1, 4)}")

z_l1 = z(w_l1, a_l0, b_l1)
a_l1 = sigmoid(z_l1)

w_l2 = 2.45
b_l2 = -0.11
print(f"w_l2 = {round(w_l2, 4)}")
print(f"b_l2 = {round(b_l2, 4)}")

z_l2 = z(w_l2, a_l1, b_l2)
a_l2 = sigmoid(z_l2)
print(f"a_l2 = {round(a_l2, 4)}")

dc_da_l2 = 2 * (a_l2-1)
dc_db_l2 = dc_db(z_l2, dc_da_l2)
dc_dw_l2 = dc_dw(a_l1, dc_db_l2)
dc_da_l1 = dc_da_prev(w_l2, dc_db_l2)

step_size = 0.1
updated_b_l2 = b_l2 - dc_db_l2 * step_size
updated_w_l2 = w_l2 - dc_dw_l2 * step_size

dc_db_l1 = dc_db(z_l1, dc_da_l1)
dc_dw_l1 = dc_dw(a_l0, dc_db_l1)

updated_b_l1 = b_l1 - dc_db_l1 * step_size
updated_w_l1 = w_l1 - dc_dw_l1 * step_size

print(f"updated_w_l1 = {round(updated_w_l1, 4)}")
print(f"updated_b_l1 = {round(updated_b_l1, 4)}")

print(f"updated_w_l2 = {round(updated_w_l2, 4)}")
print(f"updated_b_l2 = {round(updated_b_l2, 4)}")

updated_z_l1 = z(updated_w_l1, a_l0, updated_b_l1)
updated_a_l1 = sigmoid(updated_z_l1)
updated_z_l2 = z(updated_w_l2, updated_a_l1, updated_b_l2)
updated_a_l2 = sigmoid(updated_z_l2)
print(f"updated_a_l2 = {round(updated_a_l2, 4)}")
print("")
