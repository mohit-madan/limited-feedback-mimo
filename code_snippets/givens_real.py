import math
import numpy as np
import random

def givens_rot(a, b):
	if(b == 0):
		c = 1
		s = 0
	# if(abs(b) > abs(a)):
	else:	
		r = a/b
		s = 1/math.sqrt(1 + r**2)
		c = s*r
	# else:
	# 	r = b/a
	# 	c = 1/math.sqrt(1 + r**2)
	# 	s = c*r

	return c,s
	
def qr_givens(A):
	m, n = np.shape(A)
	# Q = np.identity(m, dtype = float)
	z = np.identity(n)
	I = np.zeros((m,n))
	I[:n,:] = z
	# print I
	R = A
	g = []

	for j in range(n):
		for i in range(m-1, j, -1):
			G = np.identity(m)
			c,s = givens_rot(R[i-1,j], R[i,j])
			G[i-1:i+1, i-1:i+1] = [[c,-s],[s,c]]
			g.append(G)
			R = np.matmul(np.transpose(G),R)
			# Q = np.matmul(Q,G)
	# return Q,R, g, I
	return g, I		

def given_recons(g, I):
	c = np.identity(np.shape(I)[0])
	
	for i in range (len(g)):
		c = np.matmul(c,g[i])
	
	c = np.matmul(c,I)
	return c
	
# random.seed()
x = np.random.rand(4,3)
x, _ = np.linalg.qr(x)

print x
g,I = qr_givens(x)

out = given_recons(g,I)

print out


