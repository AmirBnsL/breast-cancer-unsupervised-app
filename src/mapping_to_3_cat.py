import numpy as np

# ---------- TRAIN ----------
print(f"Original Labels (train): {np.unique(y_train)}")

y_train_3 = np.copy(y_train)

# 0: benign / low-risk (0,1,2)
benign_mask   = (y_train == 0) | (y_train == 1) | (y_train == 2)
# 1: high-risk / precursor (3,4)
highrisk_mask = (y_train == 3) | (y_train == 4)
# 2: carcinoma (5,6)
carc_mask     = (y_train == 5) | (y_train == 6)

y_train_3[benign_mask]   = 0
y_train_3[highrisk_mask] = 1
y_train_3[carc_mask]     = 2

print("3-class Labels Created (train). Counts:")
unique, counts = np.unique(y_train_3, return_counts=True)
print(dict(zip(["Benign (0)", "HighRisk (1)", "Carcinoma (2)"], counts)))

# ---------- TEST ----------
print(f"\nOriginal Labels (test): {np.unique(y_test)}")

y_test_3 = np.copy(y_test)

benign_mask   = (y_test == 0) | (y_test == 1) | (y_test == 2)
highrisk_mask = (y_test == 3) | (y_test == 4)
carc_mask     = (y_test == 5) | (y_test == 6)

y_test_3[benign_mask]   = 0
y_test_3[highrisk_mask] = 1
y_test_3[carc_mask]     = 2

print("3-class Labels Created (test). Counts:")
unique, counts = np.unique(y_test_3, return_counts=True)
print(dict(zip(["Benign (0)", "HighRisk (1)", "Carcinoma (2)"], counts)))
