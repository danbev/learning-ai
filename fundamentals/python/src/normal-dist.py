import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal_dist(mean=0, std_dev=1):
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f'Normal Distribution - Mean: {mean}, Std Dev: {std_dev}')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    while True:
        print("\nEnter the mean and standard deviation for the normal distribution.")
        try:
            mean_input = float(input("Mean (μ): "))
            std_dev_input = float(input("Standard Deviation (σ): "))
            if std_dev_input <= 0:
                print("Standard deviation must be greater than 0.")
                continue
            plot_normal_dist(mean_input, std_dev_input)
        except ValueError:
            print("Please enter valid numbers.")
        
        cont = input("Plot another? (y/n): ")
        if cont.lower() != 'y':
            break

