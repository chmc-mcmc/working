import jax.numpy as jnp
import numpy as np
import sys
sys.path.append(".")
from sampler import hmc
from jaxbugs import dnorm, dgamma

x = np.array([1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0,   8.0, 8.5, 9.0, 9.5, 9.5, 10.0, 12.0, 12.0, 13.0,   13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5])
Y = np.array([1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,   2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57])
n = 27

def U(q):
    a, b,r,t = q
    return jnp.sum(dnorm(Y,a-b*jnp.pow(r,x),t)) + dnorm(a,0,1e-6) + dnorm(b,0,1e-6) + dgamma(t,0.001,0.001)

D1_U = jax.jit(jax.grad(U))
D2_U = jax.jit(jax.hessian(U))
Uq = lambda x: D1_U(x)
Uqq = lambda x: D2_U(x)

Dim = 4
BURNIN = 2000
ITERATIONS =4000
PARTICLES = 3
outbnd=lambda q:q[2]<0.5 or q[2]>1 or q[3]<=0
qinit = np.random.uniform(0.5, 1, (PARTICLES, Dim))
QS = hmc(U, Uq,Uqq,  Dim, BURNIN, ITERATIONS,PARTICLES,outbnd=outbnd,qinit=qinit)
np.savetxt('dugongs.csv', QS, delimiter=',')
