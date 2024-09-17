import numpy as np
list1 = [49, 23, 63, 49, 18, 63]
list1 = [74, 43, 48, 8, 67, 46]
list1 = [74, 43, 48, 8]
list2 = [40, 60, 62, 46, 50, 68]
list2 = [40, 60, 62, 46]

# Step 1: Calculate deviations
deviations = (np.array(list1) - np.array(list2))**2
print(deviations)

# Step 2: Correct for variances
print(list1 + list2)
var = np.var(list1 + list2)
print(var)
corrected_deviations = deviations / var

# Step 3: Calculate the arithmetic average
arithmetic_average = np.mean(corrected_deviations)

print(f"Deviations: {deviations}")
print(f"Corrected Deviations: {corrected_deviations}")
print(f"Arithmetic Average: {arithmetic_average}")
