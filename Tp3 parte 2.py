#!/usr/bin/env python
# coding: utf-8

# In[6]:


#inicio de el punto 3-a
import numpy as np
max =50
eps =10**(-5)
a =np.array([[2,-1,0],[1,6,-2],[4,-3,8]])
b=np.array([2,-4,5])
x0=np.array([0,0,0])
iteraciones = 0
error = 2*eps
n = len(a)
x1 = np.zeros(n)
while (iteraciones <= max and error > eps): 
    for i in range(n):
        sumar = 0
        for j in range(i):
            sumar = sumar + a[i][j]*x1[j] #la primera suma
        for j in range(i+1,n):
            sumar = sumar + a[i][j]*x0[j] # la segunda suma
        x1[i] = (b[i]-sumar)/a[i][i] #el X
    error = np.abs(x1-x0).max()
    x0 = x1.copy()
    iteraciones = iteraciones + 1
if iteraciones > max:
    print("No converge en ",iteraciones)
else:
    print("CONVERGE")
    print("Iteracion ",iteraciones)
    print("Error: ",error)
    print("Aprox: ",x1)


# In[2]:


#Punto 3 Leer los archivos de texto
import numpy as np
max =50
eps =10**(-5)
a =np.loadtxt("OneDrive//Escritorio//Sistema10x10//A.txt",delimiter=",",dtype=float)
b=np.loadtxt("OneDrive//Escritorio//Sistema10x10//b.txt",delimiter=",",dtype=float)
x0=np.loadtxt("OneDrive//Escritorio//Sistema10x10//X.txt",delimiter=",",dtype=float)
iteraciones = 0
error = 2*eps
n = len(a)
x1 = np.zeros(n)
while (iteraciones <= max and error > eps): 
    for i in range(n):
        sumar = 0
        for j in range(i):
            sumar = sumar + a[i][j]*x1[j] #la primera suma
        for j in range(i+1,n):
            sumar = sumar + a[i][j]*x0[j] # la segunda suma
        x1[i] = (b[i]-sumar)/a[i][i] #el X
    error = np.abs(x1-x0).max()
    x0 = x1.copy()
    iteraciones = iteraciones + 1
if iteraciones > max:
    print("No converge en ",iteraciones)
else:
    print("CONVERGE")
    print("Iteracion ",iteraciones)
    print("Error: ",error)
    print("Aprox: ",x1)
    
#Lo que cambiaria para implementar el metodo de Jacobi seria eliminar el primer recorrido, para poder asi utilizar solo el vector inicial


# In[ ]:





# In[3]:


#punto 1 graficas
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1,10,100)
y1 = (5-2*x)/3
y2 = (5.1-2*x)/3.1

y3 = (5.001-2*x)/3
y4 = (5.1-2*x)/3.1

plt.plot(x,y1,x,y2,x,y3,x,y4)


plt.title("Graficas")
plt.legend(('y1','y2','y3','y4'), loc='upper left')
plt.xlabel('x')
plt.ylabel('y', fontsize=20)
plt.show()

# Ambas rectas son casi paralelas.Hay una zona "problematica" en la que pareciese que que las rectas son las mismas. Esto causará
# que en si en uno de los datos hay un error, voy a esta
#voy a estar muy cerca de estar en una interseccion. No es facil ver un resultado.
# Toda esa parte que coincide, son posibles resultados. Intrinsicamente, el sistema está mal condicionado. Entonces calculamos 
# k(A).
x = np.linspace(1,10,100)
y1 = (2-2*x)/3
y2 = (1.999-1.999*x)/3

y3 = (2-2*x)/3
y4 = (2-1.999*x)/3

plt.plot(x,y1,x,y2,x,y3,x,y4)


plt.title("Graficas")
plt.legend(('y1','y2','y3','y4'), loc='upper left')
plt.xlabel('x')
plt.ylabel('y', fontsize=20)
plt.show()


# In[6]:


#metodo de SOR
import numpy as np
max = 250
eps = 10**(-5)
a =np.array([[4.,3.,0.],[3.,4.,-1.],[0.,-1.,4.]])
b=np.array([24.,30.,24.])
x0=np.array([1.,2.,3.])
iteraciones = 0
error = 2*eps
n = len(a)
x1 = np.zeros(n)
w = 1.3#cagadita para que converga mas rapido
while (iteraciones <= max and error > eps): 
    for i in range(n):
        sumar = 0
        for j in range(i):
            sumar = sumar + a[i][j]*x1[j] #primera suma
        for j in range(i+1,n):
            sumar = sumar + a[i][j]*x0[j]#segunda suma
        x1[i] = (1-w) * x0[i] + w*((b[i]-sumar)/a[i,i])
    error = np.abs(x1-x0).max()
    x0 = x1.copy()
    iteraciones = iteraciones + 1
if iteraciones > max:
    print("No converge en ",iteraciones)
else:
    print("CONVERGE")
    print("Iteracion ",iteraciones)
    print("Error: ",error)
    print("Aprox: ",x1)

