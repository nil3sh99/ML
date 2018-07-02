import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

x = [1,2,3,4,5]
y = [1,2,3,4,5]
z = [5,6,7,8,9]

fig = plt.figure()

plt1 = fig.add_subplot(231)
plt1.set_xlim(0,10)
plt1.set_ylim(0,10)
plt1.set_title('Graph:')
plt1.set_xlabel('X values -->')
plt1.set_ylabel('Y values -->')
plt1.plot(x,y, '--g', label = 'City Index')
plt1.plot(x,z, '-r', label = 'expenditureIndex')
'''plt1.legend()'''
plt1.show()
