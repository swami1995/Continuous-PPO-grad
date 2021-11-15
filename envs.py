import numpy as np
import torch
import math
from numpy import sin, cos, pi
import ipdb
from matplotlib import pyplot as plt 

class PendulumDynamics(torch.nn.Module):
    def __init__(self,batch_size,device):
        super().__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.batch_size = batch_size
        self.device = device
        
    def forward(self, state, action, return_costs=False):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = action
        # u = torch.clamp(u, -2, 2)

        thetadoubledot = (-3 * g / (2 * l) * torch.sin(th + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt
        newth = th + newthdot * dt
        # newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        statedot = torch.cat((newthdot, thetadoubledot), dim=1)
        if return_costs:
            costs = 0.5*(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.05 * (u ** 2))/20
            return (state, -costs.squeeze(dim=-1), None)
        return state

    def forward_midpoint(self, state, action, return_costs=False):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = action
        u = torch.clamp(u, -2, 2)

        thetadoubledot = (-3 * g / (2 * l) * torch.sin(th + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt*0.5
        newth = th + newthdot * dt * 0.5
        thetadoubledot = (-3 * g / (2 * l) * torch.sin(newth + math.pi) + 3. / (m * l ** 2) * u)
        newthdot = thdot + thetadoubledot * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        statedot = torch.cat((newthdot, thetadoubledot), dim=1)
        if return_costs:
            costs = 0.5*(angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.05 * (u ** 2))/20
            return (state, -costs, _)
        return state

    def step(self, action, state=None, return_costs=True):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if state is None:
            state, rewards, _ = self.forward(self.state, action, return_costs=True)
        else:
            state, rewards, _ = self.forward(state, action, return_costs=True)
        self.state = state
        self.last_u = action  # for rendering
        return state, rewards, None, None
    
    def reset(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if idxs is None:
            # ipdb.set_trace()
            self.state = torch.rand((self.batch_size, 2)).to(self.device)*2 - 1
            self.state[:,0]*= np.pi#*0.0#01
            self.state[:,1]*= 8
            self.last_u = None
        else:
            # ipdb.set_trace()
            self.state[idxs] = torch.rand((sum(idxs), 2)).to(self.device)*2 - 1
            self.state[idxs,0]*=np.pi
        return self.state

    def get_frame(self, x, ax=None):
        l = self.l
        if len(x) == 2:
            th = x[0]
            cos_th = np.cos(th)
            sin_th = np.sin(th)
        elif len(x) == 3:
            cos_th, sin_th= x[0], x[1]
            th = np.arctan2(sin_th, cos_th)
        x = sin_th*l
        y = cos_th*l

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        ax.plot((0,x), (0, y), color='k')
        ax.set_xlim((-l*1.2, l*1.2))
        ax.set_ylim((-l*1.2, l*1.2))
        return fig, ax

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)



class AcrobotEnv(torch.nn.Module):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    dt = 0.05

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = torch.tensor([-1.0, 0.0, +1])

    torque_noise_max = 0.0

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 1
    # batch_size = 512

    num_obs = 4
    num_act = 1

    def __init__(self, batch_size=64, device=torch.device('cpu'), continuous=True):
        super().__init__()
        self.viewer = None
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        self.continuous = continuous
        self.actions_disc = torch.arange(-6,7,3.0).unsqueeze(-1).to(device)
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.action_space = spaces.Discrete(3)
        self.state = None
        self.device = device
        self.batch_size = batch_size
        self.gs = torch.zeros((1,4)).to(device)
        self.gs[0,0] += np.pi
        # self.seed()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def reset(self):
    #     self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,)).astype(
    #         np.float32
    #     )
    #     return self._get_ob()
    def forward1(self, s, a, return_sdot=False, un_state=False):
        if not self.continuous:
            torque = self.actions_disc[a]
        else:
            # torque = torch.round(a)
            torque = a
            # torque = torch.round(a*4)/4
            # torque = torch.clip(torque, -1, 1)


        # # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     torque += self.np_random.uniform(
        #         -self.torque_noise_max, self.torque_noise_max
        #     )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        # ipdb.set_trace()
        s_augmented = torch.cat([s, torque], dim=1)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[:, :2] = angle_normalize(ns[:, :2])
        # ns[:, 1] = angle_normalize(ns[:, 1])
        ns[:, 2] = torch.clamp(ns[:, 2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[:, 3] = torch.clamp(ns[:, 3], -self.MAX_VEL_2, self.MAX_VEL_2)
        terminal = self._terminal(ns)
        # reward = -torch.ones_like(ns[:, :1])*(1-terminal)
        coeff = [1,0.1] #[4,1] 
        # ipdb.set_trace()
        ns_dash = ns.clone()
        ns_dash[:,0] = angle_denormalize(ns[:, 0])
        gs_dash = self.gs.clone()
        gs_dash[:,0] *=0
        reward = (- coeff[0]*((ns_dash-gs_dash)*(ns-gs_dash)).sum(dim=-1) - coeff[1]*(a*a).sum(dim=-1))/100
        return ns, reward, terminal

    def forward(self, s, a, return_costs=False, un_state=False):

        if not self.continuous:
            torque = self.actions_disc[a]
        else:
            # torque = torch.round(a)*3
            torque = a
            # torque = a
            # torque = torch.round(a*4)/4
            # torque = torch.clip(torque, -6, 6)

        # ipdb.set_trace()
        # # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     torque += self.np_random.uniform(
        #         -self.torque_noise_max, self.torque_noise_max
        #     )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        # ipdb.set_trace()
        s_augmented = torch.cat([s, torque], dim=1)

        # ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns = rk4(self._dynamics, s_augmented, [0, self.dt])
        # ipdb.set_trace()
        ns_th = angle_normalize(ns[:, :2])
        # ns[:, 1] = angle_normalize(ns[:, 1])
        ns_vel2 = torch.clamp(ns[:, 2].unsqueeze(1), -self.MAX_VEL_1, self.MAX_VEL_1)
        ns_vel3 = torch.clamp(ns[:, 3].unsqueeze(1), -self.MAX_VEL_2, self.MAX_VEL_2)
        ns = torch.cat([ns_th, ns_vel2, ns_vel3], dim=1)
        terminal = self._terminal(ns)
        # reward = -torch.ones_like(ns[:, :1])*(1-terminal)
        coeff = [1,0.01] #[4,1] 
        # ipdb.set_trace()

        # ns_dash = ns.clone()
        # ns_dash[:,0] = angle_denormalize(ns[:, 0])
        # gs_dash = self.gs.clone()
        # gs_dash[:,0] *=0
        # reward = (- coeff[0]*((ns_dash-gs_dash)*(ns-gs_dash)).sum(dim=-1))# - coeff[1]*(a*a).sum(dim=-1))#/100
        reward = torch.sin(ns[:, 0]) + torch.sin(ns[:, 0] + ns[:, 1])
        return ns, reward, terminal

    def step(self, action, state=None, return_costs=True):
        if state is None:
            ns, reward, terminal = self.forward(self.state, action, return_costs=True)
        else:
            ns, reward, terminal = self.forward(state, action, return_costs=True)
        self.state = ns
        return (ns, reward, terminal, {})

    def reset(self, idxs=None):
        # TODO: Need to give _get_obs() = [costheta, sintheta, thetadot] as output
        if idxs is None:
            # ipdb.set_trace()
            self.state = torch.rand((self.batch_size, 4)).to(self.device)*2 - 1
            self.state[:,0]*= np.pi
            self.state[:,1]*= np.pi
            self.state[:,2]*= self.MAX_VEL_1
            self.state[:,3]*= self.MAX_VEL_2
            self.last_u = None
        else:
            # ipdb.set_trace()
            self.state[idxs] = torch.rand((sum(idxs), 4)).to(self.device)*2 - 1
            self.state[idxs,0]*= np.pi
            self.state[idxs,1]*= np.pi
            self.state[idxs,2]*= self.MAX_VEL_1
            self.state[idxs,3]*= self.MAX_VEL_2
        return self.state

    def _get_ob(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        )

    def _terminal(self, s):
        # s = self.state
        return (-torch.cos(s[:, 0]) - torch.cos(s[:, 1] + s[:, 0]) > 1.0).float().unsqueeze(-1)

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[:, -1]
        s = s_augmented[:, :-1]
        theta1 = s[:, 0]
        theta2 = s[:, 1]
        dtheta1 = s[:, 2]
        dtheta2 = s[:, 3]
        d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2
            ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2, ddtheta2*0.], dim=1)


    def _dynamics(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        i1 = self.LINK_MOI
        i2 = self.LINK_MOI
        g = 9.8
        s0 = s_augmented[:, :-1]
        act = s_augmented[:, -1]

        tau = act#.item()
        th1 = s0[:, 0]
        th2 = s0[:, 1]
        th1d = s0[:, 2]
        th2d = s0[:, 3]
        g = 9.8
        # ipdb.set_trace()
        TAU = torch.stack([torch.zeros_like(tau),tau], dim=1).unsqueeze(-1)

        m11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*torch.cos(th2)) + i1 + i2
        m22 = m2*lc2**2 + i2
        m12 = m2*(lc2**2 + l1*lc2*torch.cos(th2)) + i2
        M = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m12, m22*torch.ones_like(m12)], dim=-1)], dim=-2)

        h1 = -m2*l1*lc2*torch.sin(th2)*th2d**2 - 2*m2*l1*lc2*torch.sin(th2)*th2d*th1d
        h2 = m2*l1*lc2*torch.sin(th2)*th1d**2
        H = torch.stack([h1,h2],dim=-1).unsqueeze(-1)

        phi1 = (m1*lc1+m2*l1)*g*torch.cos(th1) + m2*lc2*g*torch.cos(th1+th2)
        phi2 = m2*lc2*g*torch.cos(th1+th2)
        PHI = torch.stack([phi1, phi2], dim=-1).unsqueeze(-1)

        d2th = torch.linalg.solve(M,(TAU - H - PHI)).squeeze()
        return torch.stack([th1d, th2d, d2th[:,0], d2th[:,1], th1d*0], dim=1)

    def sample_goal_state(self,):
        state = torch.rand((self.batch_size, 4)).to(self.device)*0.002 - 0.001
        state[:,0] += math.pi/2
        # mask = (state[:,0]>0).float()
        # state[:,0] = mask*(math.pi - state[:,0]) + (1-mask)*(-math.pi - state[:,0])
        return state

    def get_frame(self, s, ax=None):
        # l = self.l
        # if len(x) == 2:
        #     th = x[0]
        #     cos_th = np.cos(th)
        #     sin_th = np.sin(th)
        # elif len(x) == 3:
        #     cos_th, sin_th= x[0], x[1]
        #     th = np.arctan2(sin_th, cos_th)
        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]
        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2

        if ax is None:
            fig, ax = plt.subplots(figsize=(2*bound,2*bound))
        else:
            fig = ax.get_figure()

        ax.plot((0,p1[0]), (0, p1[1]), color='k')
        ax.plot((p1[0],p2[0]), (p1[1], p2[1]), color='k')
        ax.set_xlim((-bound*1.1, bound*1.1))
        ax.set_ylim((-bound*1.1, bound*1.1))
        return fig, ax

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

def angle_denormalize(x):
    mask = (x>0).float()
    x = (math.pi - x)*mask + (-math.pi-x)*(1-mask)
    return x

def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function

    Example 1 ::
        ## 2D system
        def derivs(x):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    This would then require re-adding the time variable to the signature of derivs.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    # try:
    #     Ny = len(y0)
    # except TypeError:
    #     yout = np.zeros((len(t),), np.float_)
    # else:
    #     yout = np.zeros((len(t), Ny), np.float_)

    # yout = torch.zeros((len(t),y0.shape[0], y0.shape[1])).to(y0)
    yout = []
    yout.append(y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]
        # y0.requires_grad_(True)
        k1 = derivs(y0)
        # ipdb.set_trace()
        k2 = derivs(y0 + dt2 * k1)
        k3 = derivs(y0 + dt2 * k2)
        k4 = derivs(y0 + dt * k3)
        yout.append(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
    # We only care about the final timestep and we cleave off action value which will be zero
    # ipdb.set_trace()
    return yout[-1][:, :4]