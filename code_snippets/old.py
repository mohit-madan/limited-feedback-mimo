#extracting the parameters theta and phi out of an orthonormal unitary matrix which can be used to reconstruct it again

import math
import cmath
import numpy as np
import random
import pdb

def unitary(n):
    X=(np.random.rand(n,n)+1j*np.random.rand(n,n))/np.sqrt(2)
    [Q,R]=np.linalg.qr(X)
    T=np.diag(np.diag(R)/np.abs(np.diag(R)))
    U=np.matrix(np.matmul(Q,T))
    # Verify print (np.matmul(U,U.H))
    return U    
    
def rand_stiefel(n,p):
    H=(np.random.rand(n,p)+1j*np.random.rand(n,p))/np.sqrt(2)
    U, S, V = np.linalg.svd(H,full_matrices=0)
    return U

# extracting givens rotation parameters
def givens_rot(a, b):
	if(b == 0):
		c = 1
		s = 0
	elif(abs(b) > abs(a)):
		r = a/b
		s = 1/math.sqrt(1 + r**2)
		c = s*r
	else:
		r = b/a
		c = 1/math.sqrt(1 + r**2)
		s = c*r
	return c,s

# extarcting phi and theta out of the matrix	
def qr_givens(A):
	m, n = np.shape(A)
	Q = np.identity(m, dtype = float)
	z = np.identity(n)
	I = np.zeros((m,n))
	I[:n,:] = z
	V = A
	g = []
	d = []
	g_theta = []
	d_phi = []
	for j in range(n):
		b = np.angle(V[j:,j])
		# print b
		d_phi.append(b)
		if(j>0):
			temp = np.ones(j)
			temp = np.append(temp, np.exp(np.vectorize(complex)(0,b)))
			a = np.diag(temp)
			d.append(a)
			V = np.matmul(np.transpose(np.conj(a)),V)
			Q = np.matmul(Q,a)
		else:
			a = np.diag(np.exp(np.vectorize(complex)(0,b)))
			d.append(a)
			V = np.matmul(np.transpose(np.conj(a)),V)
			Q = np.matmul(Q,a)

		for i in range(m-1, j, -1):
			G = np.identity(m)
			c,s = givens_rot(np.real(V[i-1,j]), np.real(V[i,j]))
			g_theta.append(np.arccos(c))

			G[i-1:i+1, i-1:i+1] = [[c,-s],[s,c]]
			g.append(G)
			V = np.matmul(np.transpose(G),V)	
			Q = np.matmul(Q,G)	

	return Q,V, g, I, d, g_theta, d_phi

# reconstruction of the matruix from the G and D
def given_recons(g,d,I):
	t, n= np.shape(I)
	c = np.identity(t)
	for i in range(n):
		c = np.matmul(c, d[i])
		for j in range (i*(2*t - (i+1))//2, (2*(i+1)*t-i*(i+3)-2)//2):
			c = np.matmul(c,g[j])
	return np.matmul(c,I)

inp = rand_stiefel(4,2)

#g_theta is of the form = [G(th_11),G(th_12),G(th_13),G(th_21),G(th_22),G(th_31)]
#d_phi is of the form = [D1[phi_11, phi_12, phi_13, phi_14], D2[phi_22, phi_23, phi_24], D3[phi_33, phi_34]]  
q,v, g,I, d, g_theta, d_phi = qr_givens(inp)

out = given_recons(g,d,I)
pdb.set_trace()
