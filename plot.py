import os
import re
import sys

import matplotlib.pyplot as plt

d = sys.argv[1]
os.chdir(d)

img_path = [p for p in os.listdir() if re.match(".*_.*_.*.png", p)]
img_path.sort()

print(len(img_path))

num_batch = len(img_path) // 2

fig, axs = plt.subplots(2, 4)

for i in range(2):
    for j in range(4):
        axs[i][j].imshow(plt.imread(img_path[num_batch*i+j]))

plt.savefig('summary.png')