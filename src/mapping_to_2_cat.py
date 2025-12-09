
print(f"Original Labels: {np.unique(y_train)}")
y_train_binary = np.copy(y_train)
y_train_binary[:] = 1

safe_mask = (y_train == 0) | (y_train == 1) | (y_train == 2)
y_train_binary[safe_mask] = 0

print(f"Binary Labels Created. Counts:")
unique, counts = np.unique(y_train_binary, return_counts=True)
print(dict(zip(["Safe (0)", "Unsafe (1)"], counts)))


print(f"Original Labels: {np.unique(y_test)}")
y_test_binary = np.copy(y_test)
y_test_binary[:] = 1

safe_mask = (y_test == 0) | (y_test == 1) | (y_test == 2)
y_test_binary[safe_mask] = 0

print(f"Binary Labels Created. Counts:")
unique, counts = np.unique(y_test_binary, return_counts=True)
print(dict(zip(["Safe (0)", "Unsafe (1)"], counts)))


