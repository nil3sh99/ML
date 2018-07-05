
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
x=[1,2,3,4,5]
y=[7,8,3,2,1]
z=[2,5,6,1,8]
#y=[1,2,3,4,5]
#z=[1,2,3,4,5]

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


# In[3]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
x=[1,2,3,4,5]
y=[1,2,3,4,5]
z=[5,6,7,8,9]

fig=plt.figure()

#plotting graph using lines
plt1=fig.add_subplot(231)
plt1.set_xlim(0,10)
plt1.set_ylim(0,10)
plt1.set_title('Graph')
plt1.set_xlabel('X values -->')
plt1.set_ylabel('Y values -->')
plt1.plot(x,y,'--g',label='City index')
plt1.plot(x,z,'-r',label='expenditure index')
plt1.legend()

#Plotting bar chart
plt2=fig.add_subplot(232)
plt2.set_xlim(0,10)
plt2.set_ylim(0,10)
plt2.set_title('Graph')
plt2.set_xlabel('X values -->')
plt2.set_ylabel('Y values -->')
plt2.bar(x,y,color='blue')

#Scattering data
plt3=fig.add_subplot(233)
plt3.scatter(x,y,s=80,marker='^',c='r')


#plotting data as pie chart

plt4=fig.add_subplot(234)
data=['Delhi','Mumbai','Chennai','Calcutta']
slices=[10,15,20,13]
color=['c','m','y','b']
plt4.pie(slices, labels=data,colors=color,explode=(0.2,0,0,0),autopct='%0.01f%%')
plt4.legend()


#plotting data in 3D


plt5=fig.add_subplot(224,projection='3d')
plt5.scatter(x,y,z,s=80,c='y')
for x,y,z in zip(x,y,z):
    text=str(x)+','+str(y)+','+str(z)
    plt5.text(x,y,z,text)

plt.show()

