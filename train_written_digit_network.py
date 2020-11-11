from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#load up the digit-pixel map dataset
digits = load_digits()

print(digits.data.shape)

plt.gray()
plt.matshow(digits.images[1])

plt.show()

