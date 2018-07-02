import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
x=[1,2,3,4,5]
#y=[7,8,3,2,1]
#z=[2,5,6,1,8]
y=[1,2,3,4,5]
z=[1,2,3,4,5]

plt.xlabel('x->')
plt.ylabel('y->')
plt.title('graph')
plt.xlim(0,20)
plt.ylim(0,20)
plt.plot(x,y,'--r',label='Sine Curve')
plt.plot(x,z,':b',label='Graph 2')
plt.legend(loc='upper center')
plt.show()
'''
'''
plt.scatter(x,y,s=100,c='red',marker='x')
plt.show()
'''
'''
plt.bar(x,y, color="green")
plt.show()
'''
'''
label=['Delhi','Mumbai','Calcutta', 'Chennai']
slices=[10,20,30,40]
plt.pie(slices,labels=label,explode=(0,0,0.1,0))
plt.legend()
plt.show()


fig=plt.figure()
plt1=fig.add_subplot(111,projection='3d')
plt1.scatter(x,y,z,s=100,c='r')
plt.show()
