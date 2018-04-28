import matplotlib.pyplot as plt
import numpy as np


tabledata = np.loadtxt("table_data.txt").T
data = []
label = []

label.append(0.2)
label.append(0.15)
label.append(0.1)
label.append(0.09)
label.append(0.08)
label.append(0.05)

for i in xrange(tabledata.shape[0]):
    if i != 0:
        # print tabledata[i]
        data.append(tabledata[i])

tabledata = tabledata[1:7]

print(len(tabledata))
print(len(label))
plt.boxplot(x=tabledata.T, labels=label)
plt.show()