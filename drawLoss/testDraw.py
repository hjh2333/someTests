from matplotlib import pyplot as plt


ax1=plt.subplot(1,1,1) #指定图画位置
x=[1,2,3,4,5,6,7,8]
y=[5,8,9,4,5,6,7,5]
plt.plot(x,y)
# ax2=plt.subplot(2,2,2)
# x=[1,2,3,4,5,6,7,8]
# y=[5,8,9,4,5,6,7,5]
# plt.plot(x,y,'o')
# ax3=plt.subplot(2,2,3)
# x=[1,2,3,4,5,6,7,8]
# y=[5,8,9,4,5,6,7,5]
# plt.plot(x,y,'-^r')
# ax4=plt.subplot(2,2,4)
# x=[1,2,3,4,5,6,7,8]
# y=[5,8,9,4,5,6,7,5]
# plt.plot(x,y,'sg')
plt.show()