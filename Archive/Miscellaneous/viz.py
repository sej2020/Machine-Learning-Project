import matplotlib.pyplot as plt


x = ['10','9','8','7','6','5','4','3']
y = [0.75, 0.79, 0.82, 0.84, 0.85, 0.84, 0.83, 0.81]

plt.bar(x, y)
plt.xlabel('Number of Attributes used for Training')
plt.ylabel('Accuracy')
plt.title('Does removing attributes improve or reduce accuracy?')
plt.show()