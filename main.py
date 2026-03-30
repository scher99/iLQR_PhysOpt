import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObstacleJacobians:
    def __init__(self, buffer=0.2):
        self.buffer = buffer

    def circle_constraints(self, z, obs_list):
        vals, jacobians = [],[]
        for obs in obs_list:
            if obs['type'] == 'circle':
                dx = z[0] - obs['x']
                dy = z[1] - obs['y']
                d = np.sqrt(dx**2 + dy**2)
                
                vals.append(d - (obs['r'] + self.buffer))
                if d > 0:
                    jacobians.append([dx/d, dy/d, 0.0])
                else:
                    jacobians.append([0.0, 0.0, 0.0])
        
        # FIX: Ensure empty lists return the correct 2D shape for numpy vstack
        if not jacobians:
            return np.array([]), np.zeros((0, 3))
            
        return np.array(vals), np.array(jacobians)

    def wall_constraints(self, z, wall_list):
        vals, jacobians = [], []
        for wall in wall_list:
            if wall['type'] == 'wall':
                x, y = z[0], z[1]
                xmin, xmax = wall['x'], wall['x'] + wall['w']
                ymin, ymax = wall['y'], wall['y'] + wall['h']
                
                if xmin < x < xmax and ymin < y < ymax:
                    d_left, d_right = x - xmin, xmax - x
                    d_bottom, d_top = y - ymin, ymax - y
                    min_d = min(d_left, d_right, d_bottom, d_top)
                    vals.append(-min_d - self.buffer)
                    
                    gx, gy = 0.0, 0.0
                    if min_d == d_left: gx = -1.0
                    elif min_d == d_right: gx = 1.0
                    elif min_d == d_bottom: gy = -1.0
                    else: gy = 1.0
                    jacobians.append([gx, gy, 0.0])
                else:
                    dx = max(xmin - x, 0, x - xmax)
                    dy = max(ymin - y, 0, y - ymax)
                    dist = np.sqrt(dx**2 + dy**2)
                    vals.append(dist - self.buffer)
                    
                    if dist > 0:
                        gx = -dx/dist if x < xmin else (dx/dist if x > xmax else 0.0)
                        gy = -dy/dist if y < ymin else (dy/dist if y > ymax else 0.0)
                        jacobians.append([gx, gy, 0.0])
                    else:
                        jacobians.append([0.0, 0.0, 0.0])
                        
        # FIX: Ensure empty lists return the correct 2D shape for numpy vstack
        if not jacobians:
            return np.array([]), np.zeros((0, 3))
            
        return np.array(vals), np.array(jacobians)


class OptimizationBasedCar:
    def __init__(self, L=2.5, dt=0.2):
        self.L, self.dt = L, dt
        self.obs =[]
        self.buffer = 0.5
        self.obs_jac = ObstacleJacobians(buffer=self.buffer)
        
        # --- NEW: Define physical limits ---
        self.v_bounds = [-5.0, 5.0]
        self.delta_bounds = [-0.3, 0.3]

    def project_controls(self, u_req):
        """ Projects requested controls to physical limits and returns the IFT gradient. """
        v_req, delta_req = u_req
        
        # 1. Project to limits
        v_app = np.clip(v_req, self.v_bounds[0], self.v_bounds[1])
        delta_app = np.clip(delta_req, self.delta_bounds[0], self.delta_bounds[1])
        u_applied = np.array([v_app, delta_app])
        
        # 2. Compute Jacobian (du_applied / du_req)
        # We use a small leaky gradient (0.01) instead of 0 to ensure the optimizer
        # still gets a tiny signal to adjust controls when saturated.
        dv = 1.0 if self.v_bounds[0] <= v_req <= self.v_bounds[1] else 0.01
        ddelta = 1.0 if self.delta_bounds[0] <= delta_req <= self.delta_bounds[1] else 0.01
        
        du_app_du_req = np.diag([dv, ddelta])
        
        return u_applied, du_app_du_req

    def _circle_constraint(self, z, obs):
        vals, _ = self.obs_jac.circle_constraints(z, [obs])
        return vals[0]

    def _circle_constraint_jac(self, z, obs):
        _, jac = self.obs_jac.circle_constraints(z, [obs])
        return jac[0]

    def _wall_constraint(self, z, obs):
        vals, _ = self.obs_jac.wall_constraints(z,[obs])
        return vals[0]

    def _wall_constraint_jac(self, z, obs):
        _, jac = self.obs_jac.wall_constraints(z, [obs])
        return jac[0]

    def get_all_obstacle_data(self, z):
        c_vals, c_G = self.obs_jac.circle_constraints(z, self.obs)
        w_vals, w_G = self.obs_jac.wall_constraints(z, self.obs)
        
        # Safely concatenate now that we know c_G and w_G are always (N, 3) 2D arrays
        vals = np.concatenate([c_vals, w_vals])
        G = np.vstack([c_G, w_G])
        
        return vals, G

    def kinematics(self, x, u):
        curr_x, curr_y, theta = x
        v, delta = u
        return np.array([
            curr_x + v * np.cos(theta) * self.dt,
            curr_y + v * np.sin(theta) * self.dt,
            theta + (v / self.L) * np.tan(delta) * self.dt
        ])
    
    def lower_level_physics(self, x_t, u_req):
        # 1. Apply physical limits
        u_applied, _ = self.project_controls(u_req)
        
        # 2. Run kinematics using the CLIPPED/APPLIED controls
        x_nom = self.kinematics(x_t, u_applied)

        def obj(z): return 0.5 * np.sum((z - x_nom) ** 2)
        def obj_jac(z): return z - x_nom

        constraints =[]
        for o in self.obs:
            if o['type'] == 'circle':
                constraints.append({'type': 'ineq', 'fun': lambda z, o=o: self._circle_constraint(z, o), 'jac': lambda z, o=o: self._circle_constraint_jac(z, o)})
            elif o['type'] == 'wall':
                constraints.append({'type': 'ineq', 'fun': lambda z, o=o: self._wall_constraint(z, o), 'jac': lambda z, o=o: self._wall_constraint_jac(z, o)})

        res = minimize(obj, x_nom, jac=obj_jac, method='SLSQP', constraints=constraints, tol=1e-6)
        return res.x, res


    def get_kinematics_jacobians(self, x, u_req):
        _, _, theta = x
        dt, L = self.dt, self.L
        
        # Get physical controls and the projection gradient
        u_applied, du_app_du_req = self.project_controls(u_req)
        v, delta = u_applied
        
        # State Jacobian (fx)
        fx = np.eye(3)
        fx[0, 2] = -v * np.sin(theta) * dt
        fx[1, 2] =  v * np.cos(theta) * dt
        
        # Control Jacobian w.r.t APPLIED control
        fu_applied = np.zeros((3, 2))
        fu_applied[0, 0] = np.cos(theta) * dt
        fu_applied[1, 0] = np.sin(theta) * dt
        fu_applied[2, 0] = (1/L) * np.tan(delta) * dt
        fu_applied[2, 1] = (v/L) * (1 / np.cos(delta)**2) * dt
        
        # Chain rule: df/du_req = df/du_applied * du_applied/du_req
        fu = fu_applied @ du_app_du_req
        
        return fx, fu
    
    def get_ift_gradients(self, x_t, u_t, z_opt, res, active_tol=1e-4):
        fx, fu = self.get_kinematics_jacobians(x_t, u_t)
        c_vals, G = self.get_all_obstacle_data(z_opt)

        if c_vals.size == 0: return fx, fu
        active = np.where(c_vals < active_tol)[0]
        if active.size == 0: return fx, fu

        G_a = G[active]
        m = G_a.shape[0]
        H = np.eye(3)

        KKT = np.block([
            [H,      -G_a.T],
            # Use small regularization to simulate a smooth barrier. This ensures
            # the gradients don't strictly vanish when the car penetrates an obstacle.
            [G_a, np.eye(m) * 1e-5] 
        ])
        rhs = np.vstack([np.eye(3), np.zeros((m, 3))])

        try:
            S = np.linalg.solve(KKT, rhs)[:3, :]
        except np.linalg.LinAlgError:
            S = np.eye(3)

        return S @ fx, S @ fu


class ILQRPlanner:
    def __init__(self, car_model, horizon=50):
        self.car = car_model
        self.N = horizon

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
    
    def _straight_line_state_guess(self, x0, x_goal):
        alphas = np.linspace(0.0, 1.0, self.N + 1)
        states = np.zeros((self.N + 1, 3), dtype=float)
        states[:, 0] = x0[0] + alphas * (x_goal[0] - x0[0])
        states[:, 1] = x0[1] + alphas * (x_goal[1] - x0[1])
        dtheta = self._wrap_angle(x_goal[2] - x0[2])
        states[:, 2] = x0[2] + alphas * dtheta
        return states

    def _initialize_controls_toward_goal(self, x0, x_goal):
        dx, dy = x_goal[0] - x0[0], x_goal[1] - x0[1]
        dist = np.hypot(dx, dy)
        u = np.zeros((self.N, 2), dtype=float)
        if dist < 1e-9: return u

        dt, L = self.car.dt, self.car.L
        goal_heading = np.arctan2(dy, dx)
        heading_error = self._wrap_angle(goal_heading - x0[2])

        turn_steps = max(1, self.N // 4)
        straight_steps = max(1, self.N - turn_steps)

        v_guess = dist / (straight_steps * dt)
        if abs(v_guess) < 1e-9:
            delta_guess = 0.0
        else:
            yaw_rate = heading_error / (turn_steps * dt)
            delta_guess = np.clip(np.arctan(L * yaw_rate / v_guess), -0.6, 0.6)

        u[:turn_steps, 0] = v_guess
        u[:turn_steps, 1] = delta_guess
        u[turn_steps:, 0] = v_guess
        return u
        
    def solve(self, x0, x_goal, weights):
        u = self._initialize_controls_toward_goal(x0, x_goal)
        init_states = self._straight_line_state_guess(x0, x_goal)
        tol, max_iters = 1e-4, 100

        # We must collect the *actually* applied controls returned by forward_pass.
        states, results, u = self.forward_pass(x0, u)

        for _ in range(max_iters):
            current_cost = self.total_cost(states, u, x_goal, weights)
            k_seq, K_seq = self.backward_pass(states, u, results, x_goal, weights)

            best_alpha, best_cost = None, current_cost
            best_states, best_u, best_results = states, u.copy(), results

            for alpha in[1.0, 0.5, 0.1, 0.05]:
                # Extract the clipped/safe simulated controls as well.
                cand_states, cand_results, cand_u = self.forward_pass(x0, u, k_seq, K_seq, alpha, states)
                cand_cost = self.total_cost(cand_states, cand_u, x_goal, weights)

                if np.isfinite(cand_cost) and cand_cost < best_cost:
                    best_alpha, best_cost = alpha, cand_cost
                    best_states, best_u, best_results = cand_states, cand_u, cand_results
                    break

            if best_alpha is None: break
            
            improvement = current_cost - best_cost
            u, states, results = best_u, best_states, best_results
            if improvement < tol: break

        return states, u, K_seq, init_states

    def backward_pass(self, states, u, results, x_goal, weights):
        N = self.N
        Q, R, Qf = weights['Q'], weights['R'], weights['Qf']
        
        # Ensure we wrap the angle so standard 2Pi shifts don't blow up the terminal cost
        err_N = states[N] - x_goal
        err_N[2] = self._wrap_angle(err_N[2])
        Vx, Vxx = Qf @ err_N, Qf
        k_seq, K_seq = [None]*N, [None]*N
        
        for t in range(N-1, -1, -1):
            z_opt, res = results[t]
            A, B = self.car.get_ift_gradients(states[t], u[t], z_opt, res)
            
            err_t = states[t] - x_goal
            err_t[2] = self._wrap_angle(err_t[2])
            
            lx, lu = Q @ err_t, R @ u[t]
            Qx, Qu = lx + A.T @ Vx, lu + B.T @ Vx
            Qxx, Quu = Q + A.T @ Vxx @ A, R + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A
            
            invQuu = np.linalg.inv(Quu + np.eye(2)*1e-6)
            k_seq[t], K_seq[t] = -invQuu @ Qu, -invQuu @ Qux
            
            Vx = Qx + K_seq[t].T @ Quu @ k_seq[t] + K_seq[t].T @ Qu + Qux.T @ k_seq[t]
            Vxx = Qxx + K_seq[t].T @ Quu @ K_seq[t] + K_seq[t].T @ Qux + Qux.T @ K_seq[t]
        return k_seq, K_seq
    
    def forward_pass(self, x0, u_nom, k=None, K=None, alpha=1.0, x_old=None):
        states, results, u_applied =[np.array(x0, dtype=float, copy=True)], [],[]

        for t in range(self.N):
            u_t = np.array(u_nom[t], dtype=float, copy=True)
            if k is not None:
                dx = states[-1] - x_old[t]
                dx[2] = self._wrap_angle(dx[2]) # Prevent crazy heading leaps
                u_t = u_t + alpha * k[t] + K[t] @ dx

            # SAFETY LIMITS: Avoid blowing up the tan() kinematics
            # u_t[0] = np.clip(u_t[0], -5.0, 5.0)  
            # u_t[1] = np.clip(u_t[1], -1.2, 1.2)  
            u_applied.append(u_t)

            x_next, res = self.car.lower_level_physics(states[-1], u_t)
            states.append(np.array(x_next, dtype=float, copy=True))
            results.append((states[-1], res))

        return np.array(states), results, np.array(u_applied)

    def total_cost(self, states, u, x_goal, weights):
        cost = 0
        Q, R, Qf = weights['Q'], weights['R'], weights['Qf']
        for t in range(self.N):
            err = states[t] - x_goal
            err[2] = self._wrap_angle(err[2])
            cost += err.T @ Q @ err + u[t].T @ R @ u[t]
        err = states[-1] - x_goal
        err[2] = self._wrap_angle(err[2])
        cost += err.T @ Qf @ err
        return cost


# --- MAIN RUN SCRIPT ---
if __name__ == "__main__":
    car = OptimizationBasedCar()
    car.obs =[
        {'type': 'circle', 'x': 5, 'y': 2, 'r': 1.0},
        {'type': 'circle', 'x': 9.0, 'y': 3, 'r': 1.0},
        # {'type': 'wall', 'x': 7, 'y': 0, 'w': 0.5, 'h': 6} # Now safely supported!
    ]

    start_pose = np.array([0.0, 0.0, 0.0])
    goal_pose  = np.array([10.0, 5.0, 0.0])
    weights = {'Q': np.diag([10, 10, 0.5]), 
               'R': np.diag([0.1, 0.1]),
               'Qf': np.diag([800, 800, 20])}

    planner = ILQRPlanner(car, horizon=30)
    planned_traj, u_nom, K_seq, init_traj = planner.solve(start_pose, goal_pose, weights)

    actual_states = [start_pose]
    x_curr = start_pose
    for t in range(len(u_nom)):
        error = x_curr - planned_traj[t]
        error[2] = planner._wrap_angle(error[2])
        
        u_feedback = u_nom[t] + K_seq[t] @ error
        # Constrain feedback to prevent noise from inducing singularity
        # u_feedback[0] = np.clip(u_feedback[0], -5.0, 5.0)
        # u_feedback[1] = np.clip(u_feedback[1], -1.2, 1.2)
        
        x_next, _ = car.lower_level_physics(x_curr, u_feedback)
        x_curr = x_next + np.random.normal(0, 0.04, 3) 
        actual_states.append(x_curr)
    actual_states = np.array(actual_states)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 12); ax.set_ylim(-1, 8); ax.set_aspect('equal')

    for o in car.obs:
        if o['type'] == 'circle': ax.add_patch(patches.Circle((o['x'], o['y']), o['r'], color='red', alpha=0.3))
        if o['type'] == 'wall': ax.add_patch(patches.Rectangle((o['x'], o['y']), o['w'], o['h'], color='black', alpha=0.5))

    ax.plot(init_traj[:,0], init_traj[:,1], 'r--', label='Initial Guess (Colliding)')
    ax.plot(planned_traj[:,0], planned_traj[:,1], 'b-', label='iLQR Plan (IFT Aware)')
    ax.plot(actual_states[:,0], actual_states[:,1], 'g-', alpha=0.6, label='True Performance (Noisy)')
    ax.scatter(goal_pose[0], goal_pose[1], marker='X', color='green', s=100, label='Goal')

    ax.legend(); plt.title("Physics-Based Trajectory Optimization"); plt.show()