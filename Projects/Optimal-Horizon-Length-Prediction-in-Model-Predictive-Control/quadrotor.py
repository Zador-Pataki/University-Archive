import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos, atan, sqrt
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables


class Quadrotor(object):
  def __init__(self, Q, R, Qf):
    self.g = 9.81
    self.m = 1
    self.a = 0.25
    self.I = 0.0625
    self.Q = Q
    self.R = R
    self.Qf = Qf

    # Input limits
    self.umin = 0
    # self.umax = 5.5
    self.umax = 20

    self.n_x = 6
    self.n_u = 2
    self.S = np.zeros((6, 6))
    try:
      self.S = np.load('S_sol.npy')
      # For the SOS problem in this homework, we maximized the rho level-set 
      # where rho = 1
      self.rho = 1.0
    except:
      print("Warning: S_sol.npy does not exist. CLF-based controllers (Problem 3) will not work")    
      print("To generate S_sol.npy, please complete Problem 2 and run stability_analysis.py")

    # Set use_experimental_inputs to True to test the CLF QP boundary controller
    # Only works after Problem 3.2.b and 3.3.b are completed
    self.use_experimental_inputs = False

  def x_d(self, t):
    # Nomial state
    # return np.array([0, 0, 0, 0, 0, 0])
    return np.array([3*cos(t), 3*sin(t), atan(3*cos(t)/(9.81 - 3*sin(t))), -3*sin(t), 3*cos(t), (-3*(1 - 0.305810397553517*sin(t))**2*sin(t) + 0.09351999925184*(9.81 - 3*sin(t))*cos(t)**2)/((9.81 - 3*sin(t))*((1 - 0.305810397553517*sin(t))**2 + 0.09351999925184*cos(t)**2))])

  def u_d(self, t):
    # Nomial input
    # return np.array([self.m*self.g/2, self.m*self.g/2])
    return np.array([5.0e-15*(1000000000000.0*sqrt((90000.0*sin(t)**2 - 588600.0*sin(t) + 90000.0*cos(t)**2 + 962361.0)/(90000.0*sin(t)**2 - 588600.0*sin(t) + 962361.0))*(981.0 - 300.0*sin(t))*(729000000000000.0*sin(t)**6 - 1.430298e+16*sin(t)**5 + 1.458e+15*sin(t)**4*cos(t)**2 + 1.169268615e+17*sin(t)**4 - 1.907064e+16*sin(t)**3*cos(t)**2 - 5.0980111614e+17*sin(t)**3 + 729000000000000.0*sin(t)**2*cos(t)**4 + 9.35414892e+16*sin(t)**2*cos(t)**2 + 1.25028723733335e+18*sin(t)**2 - 4.76766e+15*sin(t)*cos(t)**4 - 2.03920446456e+17*sin(t)*cos(t)**2 - 1.63537570643202e+18*sin(t) + 7.7951241e+15*cos(t)**4 + 1.6670496497778e+17*cos(t)**2 + 8.91279760005452e+17) - 83562883710976.0*sin(t)**5*cos(t) + 5.95957500000008e+28*sin(t)**4*cos(t) - 87960930222080.0*sin(t)**3*cos(t)**3 - 3.89756205000002e+29*sin(t)**3*cos(t) + 5.95957500000003e+28*sin(t)**2*cos(t)**3 + 3.94064967394918e+15*sin(t)**2*cos(t) - 3.89756205e+29*sin(t)*cos(t)**3 + 4.1676241244445e+30*sin(t)*cos(t) + 6.37251395174999e+29*cos(t)**3 - 6.81406544346676e+30*cos(t))/(729000000000000.0*sin(t)**6 - 1.430298e+16*sin(t)**5 + 1.458e+15*sin(t)**4*cos(t)**2 + 1.169268615e+17*sin(t)**4 - 1.907064e+16*sin(t)**3*cos(t)**2 - 5.0980111614e+17*sin(t)**3 + 729000000000000.0*sin(t)**2*cos(t)**4 + 9.35414892e+16*sin(t)**2*cos(t)**2 + 1.25028723733335e+18*sin(t)**2 - 4.76766e+15*sin(t)*cos(t)**4 - 2.03920446456e+17*sin(t)*cos(t)**2 - 1.63537570643202e+18*sin(t) + 7.7951241e+15*cos(t)**4 + 1.6670496497778e+17*cos(t)**2 + 8.91279760005452e+17), 5.0e-15*(1000000000000.0*sqrt((90000.0*sin(t)**2 - 588600.0*sin(t) + 90000.0*cos(t)**2 + 962361.0)/(90000.0*sin(t)**2 - 588600.0*sin(t) + 962361.0))*(981.0 - 300.0*sin(t))*(729000000000000.0*sin(t)**6 - 1.430298e+16*sin(t)**5 + 1.458e+15*sin(t)**4*cos(t)**2 + 1.169268615e+17*sin(t)**4 - 1.907064e+16*sin(t)**3*cos(t)**2 - 5.0980111614e+17*sin(t)**3 + 729000000000000.0*sin(t)**2*cos(t)**4 + 9.35414892e+16*sin(t)**2*cos(t)**2 + 1.25028723733335e+18*sin(t)**2 - 4.76766e+15*sin(t)*cos(t)**4 - 2.03920446456e+17*sin(t)*cos(t)**2 - 1.63537570643202e+18*sin(t) + 7.7951241e+15*cos(t)**4 + 1.6670496497778e+17*cos(t)**2 + 8.91279760005452e+17) + 83562883710976.0*sin(t)**5*cos(t) - 5.95957500000008e+28*sin(t)**4*cos(t) + 87960930222080.0*sin(t)**3*cos(t)**3 + 3.89756205000002e+29*sin(t)**3*cos(t) - 5.95957500000003e+28*sin(t)**2*cos(t)**3 - 3.94064967394918e+15*sin(t)**2*cos(t) + 3.89756205e+29*sin(t)*cos(t)**3 - 4.1676241244445e+30*sin(t)*cos(t) - 6.37251395174999e+29*cos(t)**3 + 6.81406544346676e+30*cos(t))/(729000000000000.0*sin(t)**6 - 1.430298e+16*sin(t)**5 + 1.458e+15*sin(t)**4*cos(t)**2 + 1.169268615e+17*sin(t)**4 - 1.907064e+16*sin(t)**3*cos(t)**2 - 5.0980111614e+17*sin(t)**3 + 729000000000000.0*sin(t)**2*cos(t)**4 + 9.35414892e+16*sin(t)**2*cos(t)**2 + 1.25028723733335e+18*sin(t)**2 - 4.76766e+15*sin(t)*cos(t)**4 - 2.03920446456e+17*sin(t)*cos(t)**2 - 1.63537570643202e+18*sin(t) + 7.7951241e+15*cos(t)**4 + 1.6670496497778e+17*cos(t)**2 + 8.91279760005452e+17)])

  def continuous_time_full_dynamics(self, x, u):
    # Dynamics for the quadrotor
    g = self.g
    m = self.m
    a = self.a
    I = self.I

    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]
    u0 = u[0]
    u1 = u[1]

    xdot = np.array([ydot,
                     zdot,
                     thetadot,
                     -sin(theta) * (u0 + u1) / m,
                     -g + cos(theta) * (u0 + u1) / m,
                     a * (u0 - u1) / I])
    return xdot

  def continuous_time_linearized_dynamics(self,t):
    # Dynamics linearized at the fixed point
    # This function returns A and B matrix
    # A = np.zeros((6,6))
    # A[:3, -3:] = np.identity(3)
    # A[3, 2] = -self.g;

    # B = np.zeros((6,2))
    # B[4,0] = 1/self.m;
    # B[4,1] = 1/self.m;
    # B[5,0] = self.a/self.I
    # B[5,1] = -self.a/self.I

    x_d = self.x_d(t)
    u_d = self.u_d(t)
    m = self.m
    a = self.a
    I = self.I

    A = np.array([[0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1],
                  [0,0,-cos(x_d[2])/m * (u_d[0] + u_d[1]),0,0,0],
                  [0,0,-sin(x_d[2])/m * (u_d[0] + u_d[1]),0,0,0],
                  [0,0,0,0,0,0]])
    B = np.array([[0,0],
                  [0,0],
                  [0,0],
                  [-sin(x_d[2])/m, -sin(x_d[2])/m],
                  [ cos(x_d[2])/m,  cos(x_d[2])/m],
                  [a/I, -a/I]])

    return A, B

  def discrete_time_linearized_dynamics(self, T,t):
    # Discrete time version of the linearized dynamics at the fixed point
    # This function returns A and B matrix of the discrete time dynamics
    A_c, B_c = self.continuous_time_linearized_dynamics(t)
    A_d = np.identity(6) + A_c * T;
    B_d = B_c * T;

    return A_d, B_d

  def add_initial_state_constraint(self, prog, x, x_current):
    # TODO: impose initial state constraint.
    # Use AddBoundingBoxConstraint
    for i in range(6):
      prog.AddBoundingBoxConstraint(x_current[i],x_current[i],x[0,i])
    
    pass

  def add_input_saturation_constraint(self, prog, x, u, N,t):
    # TODO: impose input limit constraint.
    # Use AddBoundingBoxConstraint
    # The limits are available through self.umin and self.umax
    ud = self.u_d(t)
    for i in range(N-1):
      # remember to add to u to make it not error
      prog.AddBoundingBoxConstraint(self.umin-ud,self.umax-ud,u[i])
    pass

  def add_dynamics_constraint(self, prog, x, u, N, T,t):
    # TODO: impose dynamics constraint.
    # Use AddLinearEqualityConstraint(expr, value)
    # A, B = self.discrete_time_linearized_dynamics(T,t)

    for i in range(N-1):
      A, B = self.discrete_time_linearized_dynamics(T,t+i*T)
      x_kp1 = x[i+1] - self.x_d(t+(i+1)*T)
      x_curr = A @ (x[i]-self.x_d(t+i*T)) + B @ (u[i]) # compute what x_k+1 should be
      for q in range(6): # do that for all 6
        prog.AddLinearEqualityConstraint(x_kp1[q] == x_curr[q])
      # prog.AddLinearEqualityConstraint(x[i+1] = A @ x[i] + B @ u[i])
    pass

  def add_cost(self, prog, x, u, N, t, T):
    # TODO: add cost.
    cost = 0
    Qk = np.block([[self.Q,np.zeros((6,2))],
                   [np.zeros((2,6)), self.R]])
    #z_d = np.array([np.concatenate((self.x_d(),self.u_d()))]).T# the desired value
    for k in range(N-1):
      z_d = np.array([np.concatenate((self.x_d(t+k*T),np.zeros((2))))]).T# the desired value
      zk = np.array([np.concatenate((x[k],u[k]))]).T - z_d
      cost_k = zk.T @ Qk @ zk
      cost += cost_k
      # prog.AddQuadraticErrorCost(Qk,z_d,zk)
    QT = np.block([[self.Qf,np.zeros((6,2))],
                   [np.zeros((2,6)), self.R]])
    # zT = np.array([np.concatenate((x[-1],u[-1]))]).T
    zT = np.array([x[-1] - self.x_d(t+T*N)]).T
    # cost += (zT.T @ QT @ zT)/2
    # cost += (zT.T @ self.Qf @ zT)
    cost += (zT.T @ self.S @ zT)

    prog.AddQuadraticCost(cost[0,0])
    pass

  def add_mpc_clf_constraint(self, prog, x, N):
    '''
    Adds the constraint V(x_t) <= V(x_0)
    Note that this constraint is non-linear and turns this MPC problem from a 
    quadratic program to a non-linear program
    '''
    if (not np.allclose(self.S, np.zeros((self.n_x, self.n_x)))):
      # TODO: add the discrete time stability constraint within this if statement
      prog.AddConstraint(x[N-1].T @ self.S @ x[N-1] - x[0].T @ self.S @ x[0] <= 0)
      pass

  def compute_mpc_feedback(self, x_current, t_current,N=10,use_clf=False):
    '''
    This function computes the MPC controller input u
    '''
    
    # Parameters for the QP
    T = 0.1

    # Initialize mathematical program and declare decision variables
    prog = MathematicalProgram()
    x = np.zeros((int(N), 6), dtype="object")
    for i in range(N):
      x[i] = prog.NewContinuousVariables(6, "x_" + str(i))
    u = np.zeros((N-1, 2), dtype="object")
    for i in range(N-1):
      u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

    # Add constraints and cost
    self.add_initial_state_constraint(prog, x, x_current)
    self.add_input_saturation_constraint(prog, x, u, N, t_current)
    self.add_dynamics_constraint(prog, x, u, N, T, t_current)
    self.add_cost(prog, x, u, N, t_current, T)

    # Placeholder constraint and cost to satisfy QP requirements
    # TODO: Delete after completing this function
    # prog.AddQuadraticCost(0)
    # prog.AddLinearEqualityConstraint(0, 0)
    
    # Adds the stability constraint: V(x_T) <= V(x_0) if using 
    # the clf version of MPC
    if (use_clf) :
      self.add_mpc_clf_constraint(prog, x, N)

    # Solve the QP
    solver = OsqpSolver()
    if (use_clf) :
      # Because we've added the CLF constraint here,
      # this problem becomes a non-linear program
      solver = SnoptSolver()
    result = solver.Solve(prog)

    u_mpc = np.zeros(2)
    # TODO: retrieve the controller input from the solution of the optimization problem
    # and use it to compute the MPC input u
    # You should make use of result.GetSolution(decision_var) where decision_var
    # is the variable you want

    u_mpc = result.GetSolution(u[0])#[result.GetSolution(u[i]) for i in range(N-1) if True]
    # for i in range(N):
    #   u0 = result.GetSolution("u_" + str(i)))

    u_mpc += self.u_d(t_current) # remember to add the nominal input

    return u_mpc

  def compute_lqr_feedback(self, x):
    '''
    Infinite horizon LQR controller
    '''
    A, B = self.continuous_time_linearized_dynamics()
    S = solve_continuous_are(A, B, self.Q, self.R)
    K = -inv(self.R) @ B.T @ S
    u = self.u_d() + K @ x;
    return u

  def dynamics_cubic_approximation(self, x, u):
    '''
    Approximated Dynamics for the quadrotor.
    We substitute
      sin(theta) = theta - (theta**3)/6
      cos(theta) = 1 - (theta**2)/2
    into the full dynamics.
    '''
    g = self.g
    m = self.m
    a = self.a
    I = self.I

    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]
    u0 = u[0]
    u1 = u[1]

    xdot = np.array([ydot,
                     zdot,
                     thetadot,
                     -(theta - (theta**3)/6) * (u0 + u1) / m,
                     -g + (1 - (theta**2)/2) * (u0 + u1) / m,
                     a * (u0 - u1) / I])
    return xdot

  def closed_loop_dynamics_cubic_approximation(self, x):
    # Closed-loop dynamics with infinite horizon LQR
    u = self.compute_lqr_feedback(x)
    return self.dynamics_cubic_approximation(x, u)

  def f(self, x):
    '''
    Returns the f(x) component of the control affine dynamics xdot = f(x) + h(x) u
    '''
    # Extract individual components of x, may not need all variables
    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]

    f_x = np.zeros((6)) # was ((6,1))
    # TODO: fill in the values for f_x
    # f_x[0] = ydot
    # f_x[1] = zdot
    # f_x[2] = thetadot
    # f_x[4] = -self.g * theta
    f_x[0] = ydot
    f_x[1] = zdot
    f_x[2] = thetadot
    f_x[4] = -self.g

    return f_x

  def h(self, x):
    '''
    Returns the h(x) component of the control affine dynamics xdot = f(x) + h(x) u
    '''
    # Extract individual components of x, may not need all variables
    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]

    g_x = np.zeros((self.n_x, self.n_u))
    # TODO: fill in the values for g_x
    # g_x[4,0] = 1/self.m
    # g_x[4,1] = 1/self.m
    # g_x[5,0] =  self.a/self.I
    # g_x[5,1] = -self.a/self.I

    g_x[3,0] = -np.sin(theta)/self.m
    g_x[3,1] = -np.sin(theta)/self.m
    g_x[4,0] = np.cos(theta)/self.m
    g_x[4,1] = np.cos(theta)/self.m
    g_x[5,0] =  self.a/self.I
    g_x[5,1] = -self.a/self.I

    return g_x

  def compute_clf_qp_feedback(self, x):
    '''
    Find control input u using V such that Vdot leq 0
    by constructing a QP that minimizes u^T R u
                              s.t.   dVdx h(x) u <= -dVdx f(x) 
    '''
    if self.use_experimental_inputs and not self.is_near_boundary(x):
      return np.clip(6 * np.random.random(2), self.umin, self.umax)

    # You can retrieve the system paramters m, g, a, and I from the class variables
    # For example, a can be retrieved by calling self.a

    # Boiler plate code to construct the mathematical program
    prog = MathematicalProgram()
    u = prog.NewContinuousVariables(self.n_u, "u")
    R = np.eye(self.n_u)

    # Add cost function
    prog.AddQuadraticCost(u.T @ R @ u)

    # TODO: Add the constraint dVdx h(x) u <= -dVdx f(x)
    # This is a constraint that is linear in the decision variables u, so we can 
    # use prog.AddLinearConstraint(expr)

    prog.AddLinearConstraint(x.T @ self.S @ self.h(x) @ u + x.T @ self.S @ self.f(x) <= 0)

    solver = OsqpSolver()
    result = solver.Solve(prog)

    return result.GetSolution(u)

  def is_near_boundary(self, x, tol = 1e-2):
    '''
    Returns whether the state x is within tol of the boundary of the ROA
    '''
    near_boundary = True

    # TODO: Using V = x.T S x computed in Problem 2, set near_boundary to True if
    # V(x) is near the boundary of the ROA.
    
    V = x.T @ self.S @ x

    near_boundary = self.rho - V <= tol

    return near_boundary