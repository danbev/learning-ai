array = [2, 4, 4, 4, 5, 5, 7, 9, 12, 18]

mean = sum(array) / len(array)
std = (sum((x - mean)**2 for x in array) / len(array)) **0.5

normalized = [(x - mean) / std for x in array]

print('Original Array:', array)
print('Normalized:\n', '\n'.join(map(str, normalized)))

normalized_mean = sum(normalized) / len(normalized)
print('Normalized Mean:', normalized_mean)
