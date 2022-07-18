import logging
import os
import readline

from matplotlib import pyplot as plt

to_ckpt_dir = r'C:\Users\t-jiahuihe\Code\PythonCode\TestForFirst\drawLoss'
info_path = os.path.join(to_ckpt_dir, "loss_log.txt")
with open(info_path, 'a', encoding='utf-8')as file:
    # file.write('average\n')
    pass
draw = plt.subplot(2000,2000,1)
x = []
y = []
plt.plot(x,y)
with open(info_path, 'r', encoding='utf-8')as file:
    while 1:
        line = file.readline()
        if not line:
            break
        print(line)
        pair = line.split()
        print(pair,type(pair))
        step = pair[0]
        loss = pair[1]
        x.append(step)
        y.append(loss)
        print(step, loss)

plt.show()