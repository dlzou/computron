import csv
import numpy as np
import matplotlib.pyplot as plt


def plot(filename):


    latency=[]
    with open(filename) as f:
        reader=csv.DictReader(f)
        for row in reader:
            # print(row)
            latency.append(float(row['Response Time'])-float(row['Receive Time']))
            if float(row['Response Time'])-float(row['Receive Time'])>1:
                print(row)

    latency.sort()
    # print(latency)

    data=latency
    x = np.sort(data)

    y = 1. * np.arange(len(data)) / (len(data) - 1)

    #plot CDF
    # plt.plot(x, y)
    # plt.xlabel('Latency')
    # plt.ylabel('CDF')
    # plt.xlim((0,0.2))
    # plt.show()

    return (x,y)

(x1,y1)=plot("1.csv")
(x2,y2)=plot("2.csv")

# print(y1[-3])
# print(y2[-3])

plt.plot(x1, y1, label="computron")
plt.plot(x2, y2, label="energon")
plt.xlabel('Latency')
plt.ylabel('CDF')
plt.xlim((0,0.2))
plt.legend()
plt.show()