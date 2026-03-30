import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObstacleJacobians:
    def __init__(self, buffer=0.2):
        self.buffer = buffer

    def circle_constraints(self, z, obs_list):
        """Returns the values c(z) and Jacobians Dzc for all circles."""
        vals = []
        jacobians = []
        
        for obs in obs_list:
            if obs['type'] == 'circle':
                dx = z[0] - obs['x']
                dy = z[1] - obs['y']
                d = np.sqrt(dx**2 + dy**2)
                
                # Constraint value
                vals.append(d - (obs['r'] + self.buffer))
                
                # Jacobian: [dc/dx, dc/dy, dc/dtheta]
                if d > 0:
                    jacobians.append([dx/d, dy/d, 0.0])
                else:
                    jacobians.append([0.0, 0.0, 0.0])
        
        return np.array(vals), np.array(jacobians)

    def wall_constraints(self, z, wall_list):
        """Returns values and Jacobians for rectangular walls."""
        vals = []
        jacobians = []
        
        for wall in wall_list:
            if wall['type'] == 'wall':
                # Simple axis-aligned logic: distance to the box
                # x_dist = max(x_min - x, 0, x - x_max)
                # For brevity, we treat as a single point-to-plane check
                # based on which side of the wall the car is on.
                x, y = z[0], z[1]
                xmin, xmax = wall['x'], wall['x'] + wall['w']
                ymin, ymax = wall['y'], wall['y'] + wall['h']
                
                # Check nearest edge
                dx = max(xmin - x, 0, x - xmax)
                dy = max(ymin - y, 0, y - ymax)
                
                dist = np.sqrt(dx**2 + dy**2)
                vals.append(dist - self.buffer)
                
                # Gradient points towards the exterior
                gx = -1 if x < xmin else (1 if x > xmax else 0)
                gy = -1 if y < ymin else (1 if y > ymax else 0)
                jacobians.append([gx, gy, 0.0])
                
        return np.array(vals), np.array(jacobians)


class OptimizationBasedCar:
    def __init__(self, L=2.5, dt=0.2):
        self.L, self.dt = L, dt
        self.obs = []
        self.buffer = 0.5
        self.obs_jac = ObstacleJacobians(buffer=self.buffer)

    def _circle_constraint(self, z, obs):
        d = z[:2] - np.array([obs['x'], obs['y']])
        return np.linalg.norm(d) - (obs['r'] + self.buffer)

    def _circle_constraint_jac(self, z, obs):
        d = z[:2] - np.array([obs['x'], obs['y']])
        n = np.linalg.norm(d)
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0])
        return np.array([d[0] / n, d[1] / n, 0.0])

    def get_all_obstacle_data(self, z):
        c_vals, G = self.obs_jac.circle_constraints(z, self.obs)
        if c_vals.size == 0:
            return np.array([]), np.zeros((0, 3))
        return c_vals, G

    def kinematics(self, x, u):
        curr_x, curr_y, theta = x
        v, delta = u
        return np.array([
            curr_x + v * np.cos(theta) * self.dt,
            curr_y + v * np.sin(theta) * self.dt,
            theta + (v / self.L) * np.tan(delta) * self.dt
        ])
    
    def lower_level_physics(self, x_t, u_t):
        x_nom = self.kinematics(x_t, u_t)

        def obj(z):
            return 0.5 * np.sum((z - x_nom) ** 2)

        def obj_jac(z):
            return z - x_nom

        constraints = []
        for o in self.obs:
            if o['type'] == 'circle':
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, o=o: self._circle_constraint(z, o),
                    'jac': lambda z, o=o: self._circle_constraint_jac(z, o),
                })

        res = minimize(
            obj,
            x_nom,
            jac=obj_jac,
            method='SLSQP',
            constraints=constraints,
            tol=1e-6
        )
        return res.x, res

    def get_kinematics_jacobians(self, x, u):
        """Analytical derivatives for iLQR."""
        _, _, theta = x
        v, delta = u
        dt, L = self.dt, self.L
        fx = np.eye(3)
        fx[0, 2] = -v * np.sin(theta) * dt
        fx[1, 2] =  v * np.cos(theta) * dt
        fu = np.zeros((3, 2))
        fu[0, 0] = np.cos(theta) * dt
        fu[1, 0] = np.sin(theta) * dt
        fu[2, 0] = (1/L) * np.tan(delta) * dt
        fu[2, 1] = (v/L) * (1 / np.cos(delta)**2) * dt
        return fx, fu
    
    def get_ift_gradients(self, x_t, u_t, z_opt, res, active_tol=1e-5):
        fx, fu = self.get_kinematics_jacobians(x_t, u_t)
        c_vals, G = self.get_all_obstacle_data(z_opt)

        if c_vals.size == 0:
            return fx, fu

        active = np.where(c_vals < active_tol)[0]
        if active.size == 0:
            return fx, fu

        G_a = G[active]
        m = G_a.shape[0]
        H = np.eye(3)

        KKT = np.block([
            [H,      -G_a.T],
            [G_a, np.zeros((m, m))]
        ])
        rhs = np.vstack([
            np.eye(3),
            np.zeros((m, 3))
        ])

        try:
            S = np.linalg.solve(KKT, rhs)[:3, :]
        except np.linalg.LinAlgError:
            S = np.eye(3)

        return S @ fx, S @ fu


class SimulationWorld:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.obstacles = []  # List of dicts: {'type': 'circle', 'x', 'y', 'r'} or {'type': 'wall', ...}

    def add_random_pillars(self, n=5, min_r=0.5, max_r=1.5):
        """Adds random circular obstacles."""
        for _ in range(n):
            self.obstacles.append({
                'type': 'circle',
                'x': np.random.uniform(2, self.width - 2),
                'y': np.random.uniform(2, self.height - 2),
                'r': np.random.uniform(min_r, max_r)
            })

    def add_wall(self, x, y, w, h, angle_deg=0):
        """Adds a rectangular wall (long and narrow)."""
        self.obstacles.append({
            'type': 'wall',
            'x': x, 'y': y, 'w': w, 'h': h, 'angle': angle_deg
        })

    def plot_world(self, ax=None):
        """Visualizes the obstacles."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

        for obs in self.obstacles:
            if obs['type'] == 'circle':
                circle = patches.Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.6)
                ax.add_patch(circle)
            elif obs['type'] == 'wall':
                # Simplified rectangular wall for now (no rotation logic for plotting yet)
                rect = patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], 
                                         angle=obs['angle'], color='black', alpha=0.8)
                ax.add_patch(rect)
        
        return ax

class ILQRPlanner:
    def __init__(self, car_model, horizon=50):
        self.car = car_model
        self.N = horizon # Number of time steps

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
        dx = x_goal[0] - x0[0]
        dy = x_goal[1] - x0[1]
        dist = np.hypot(dx, dy)

        u = np.zeros((self.N, 2), dtype=float)
        if dist < 1e-9:
            return u

        dt = self.car.dt
        L = self.car.L

        goal_heading = np.arctan2(dy, dx)
        heading_error = self._wrap_angle(goal_heading - x0[2])

        turn_steps = max(1, self.N // 4)
        straight_steps = max(1, self.N - turn_steps)

        v_guess = dist / (straight_steps * dt)

        if abs(v_guess) < 1e-9:
            delta_guess = 0.0
        else:
            yaw_rate = heading_error / (turn_steps * dt)
            delta_guess = np.arctan(L * yaw_rate / v_guess)

        delta_guess = np.clip(delta_guess, -0.6, 0.6)

        u[:turn_steps, 0] = v_guess
        u[:turn_steps, 1] = delta_guess
        u[turn_steps:, 0] = v_guess
        u[turn_steps:, 1] = 0.0

        return u
        
    def solve(self, x0, x_goal, weights):
        u = self._initialize_controls_toward_goal(x0, x_goal)

        # # Save the initial guess trajectory for plotting
        # init_states, _ = self.forward_pass(x0, u.copy())
        # Straight-line initial path for visualization only
        init_states = self._straight_line_state_guess(x0, x_goal)


        K_seq = [np.zeros((2, 3)) for _ in range(self.N)]
        tol = 1e-4
        max_iters = 100

        for _ in range(max_iters):
            states, results = self.forward_pass(x0, u)
            current_cost = self.total_cost(states, u, x_goal, weights)

            k_seq, K_seq = self.backward_pass(states, u, results, x_goal, weights)

            best_alpha = None
            best_cost = current_cost
            best_states = states
            best_u = u.copy()

            # Line search on a control sequence consistent with the rollout
            for alpha in [1.0, 0.5, 0.1, 0.05]:
                cand_states, _ = self.forward_pass(x0, u, k_seq, K_seq, alpha, states)

                cand_u = np.array([
                    u[t] + alpha * k_seq[t] + K_seq[t] @ (cand_states[t] - states[t])
                    for t in range(self.N)
                ])

                cand_cost = self.total_cost(cand_states, cand_u, x_goal, weights)

                if np.isfinite(cand_cost) and cand_cost < best_cost:
                    best_alpha = alpha
                    best_cost = cand_cost
                    best_states = cand_states
                    best_u = cand_u
                    break

            # No improving step found
            if best_alpha is None:
                break

            improvement = current_cost - best_cost
            u = best_u

            if improvement < tol:
                break

        # Recompute final rollout and gains so outputs are consistent
        final_states, final_results = self.forward_pass(x0, u)
        _, K_seq = self.backward_pass(final_states, u, final_results, x_goal, weights)

        return final_states, u, K_seq, init_states


    def backward_pass(self, states, u, results, x_goal, weights):
        N = self.N
        Q, R, Qf = weights['Q'], weights['R'], weights['Qf']
        Vx = Qf @ (states[N] - x_goal)
        Vxx = Qf
        k_seq, K_seq = [None]*N, [None]*N
        
        for t in range(N-1, -1, -1):
            z_opt, res = results[t] # Correct unpacking
            A, B = self.car.get_ift_gradients(states[t], u[t], z_opt, res)
            
            lx, lu = Q @ (states[t] - x_goal), R @ u[t]
            Qx, Qu = lx + A.T @ Vx, lu + B.T @ Vx
            Qxx, Quu = Q + A.T @ Vxx @ A, R + B.T @ Vxx @ B
            Qux = B.T @ Vxx @ A
            
            invQuu = np.linalg.inv(Quu + np.eye(2)*1e-6)
            k_seq[t], K_seq[t] = -invQuu @ Qu, -invQuu @ Qux
            
            Vx = Qx + K_seq[t].T @ Quu @ k_seq[t] + K_seq[t].T @ Qu + Qux.T @ k_seq[t]
            Vxx = Qxx + K_seq[t].T @ Quu @ K_seq[t] + K_seq[t].T @ Qux + Qux.T @ K_seq[t]
        return k_seq, K_seq
    
    def forward_pass(self, x0, u_nom, k=None, K=None, alpha=1.0, x_old=None):
        states = [np.array(x0, dtype=float, copy=True)]
        results = []

        for t in range(self.N):
            u_t = np.array(u_nom[t], dtype=float, copy=True)

            if k is not None:
                u_t = u_t + alpha * k[t] + K[t] @ (states[-1] - x_old[t])

            x_next, res = self.car.lower_level_physics(states[-1], u_t)
            x_next = np.array(x_next, dtype=float, copy=True)

            states.append(x_next)
            results.append((x_next, res))

        return np.array(states), results

    def calculate_total_cost(self, states, u_seq, x_goal, weights):
        """Computes the scalar cost of the current trajectory."""
        total_cost = 0
        Q, R, Qf = weights['Q'], weights['R'], weights['Qf']
        
        for t in range(self.N):
            state_err = states[t] - x_goal
            total_cost += state_err.T @ Q @ state_err + u_seq[t].T @ R @ u_seq[t]
            
        final_err = states[-1] - x_goal
        total_cost += final_err.T @ Qf @ final_err
        return total_cost

    def line_search(self, x0, u_old, k_seq, K_seq, old_cost, x_goal, weights):
        """Finds a step size alpha that actually reduces the cost."""
        alphas = [1.0, 0.5, 0.25, 0.1]
        for alpha in alphas:
            u_new = []
            x_curr = x0
            for t in range(self.N):
                # Apply feedback law: u = u_nominal + alpha*k + K*(x - x_nominal)
                # (For simplicity here, we use a basic update)
                u_t = u_old[t] + alpha * k_seq[t] 
                u_new.append(u_t)
                
            new_states, _ = self.forward_rollout(x0, u_new)
            new_cost = self.calculate_total_cost(new_states, u_new, x_goal, weights)
            
            if new_cost < old_cost:
                return np.array(u_new), new_cost
        return u_old, old_cost
    
    def total_cost(self, states, u, x_goal, weights):
        cost = sum((states[t]-x_goal).T @ weights['Q'] @ (states[t]-x_goal) + u[t].T @ weights['R'] @ u[t] for t in range(self.N))
        return cost + (states[-1]-x_goal).T @ weights['Qf'] @ (states[-1]-x_goal)

def plot_results(world, initial_states, optimized_states, x_goal):
    fig, ax = plt.subplots(figsize=(10, 8))
    world.plot_world(ax)
    
    # Plot Initial Guess (Red dotted)
    ax.plot(initial_states[:, 0], initial_states[:, 1], 'r--', label='Initial Guess', alpha=0.5)
    
    # Plot Optimized Solution (Blue solid)
    ax.plot(optimized_states[:, 0], optimized_states[:, 1], 'b-o', label='iLQR Solution', markersize=3)
    
    # Plot Goal
    ax.plot(x_goal[0], x_goal[1], 'gx', markersize=10, mew=3, label='Goal')
    
    # Add orientation arrows for the final states
    for s in [initial_states[-1], optimized_states[-1]]:
        ax.arrow(s[0], s[1], 0.5*np.cos(s[2]), 0.5*np.sin(s[2]), head_width=0.2, color='black')
        
    ax.legend()
    plt.title("Maneuver Planning with Optimization-Based Dynamics")
    plt.show()

def run_noisy_simulation(car, x0, u_nominal, K_seq, x_nominal, noise_std=0.05):
    """
    Simulates the car executing the plan with real-time feedback and random noise.
    """
    x_curr = x0
    actual_states = [x0]
    
    for t in range(len(u_nominal)):
        # 1. Calculate feedback control: u = u_nom + K * (x_actual - x_nom)
        error = x_curr - x_nominal[t]
        u_feedback = u_nominal[t] + K_seq[t] @ error
        
        # 2. Step the physics
        x_next, _ = car.lower_level_physics(x_curr, u_feedback)
        
        # 3. Apply Gaussian noise to the state (e.g., wind or sensor error)
        noise = np.random.normal(0, noise_std, size=3)
        x_curr = x_next + noise
        actual_states.append(x_curr)
        
    return np.array(actual_states)
   
# # --- Setup the World ---
# world = SimulationWorld(width=20, height=15)

# # Add some pillars
# np.random.seed(42) # For reproducible random obstacles
# world.add_random_pillars(n=4)

# # Add a 'Wall' (a narrow corridor or barrier)
# world.add_wall(x=10, y=2, w=1, h=8)  # Vertical wall in the middle

# # Visualize the result
# ax = world.plot_world()
# ax.set_title("Optimization-Based Dynamics: Simulation World")
# plt.show()
# 1. Setup World
car = OptimizationBasedCar()
car.obs = [
    {'type': 'circle', 'x': 5, 'y': 2, 'r': 1.0},
    {'type': 'circle', 'x': 9.0, 'y': 3, 'r': 1.0},
    # {'type': 'wall', 'x': 7, 'y': 0, 'w': 0.5, 'h': 6}
]

# 2. Define Maneuver
start_pose = np.array([0.0, 0.0, 0.0])
goal_pose  = np.array([10.0, 5.0, 0.0])
weights = {'Q': np.diag([10, 10, 0.5]), 
           'R': np.diag([0.1, 0.1]),
           'Qf': np.diag([8000, 8000, 200])}


# 3. Plan
planner = ILQRPlanner(car, horizon=30)
planned_traj, u_nom, K_seq, init_traj = planner.solve(start_pose, goal_pose, weights)

# 4. Noisy Simulation (True Performance)
actual_states = [start_pose]
x_curr = start_pose
for t in range(len(u_nom)):
    u_feedback = u_nom[t] + K_seq[t] @ (x_curr - planned_traj[t])
    x_next, _ = car.lower_level_physics(x_curr, u_feedback)
    x_curr = x_next + np.random.normal(0, 0.04, 3) # Added Noise
    actual_states.append(x_curr)
actual_states = np.array(actual_states)

# 5. Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1, 12); ax.set_ylim(-1, 8); ax.set_aspect('equal')

# Draw Obstacles
for o in car.obs:
    if o['type'] == 'circle': ax.add_patch(patches.Circle((o['x'], o['y']), o['r'], color='red', alpha=0.3))
    if o['type'] == 'wall': ax.add_patch(patches.Rectangle((o['x'], o['y']), o['w'], o['h'], color='black', alpha=0.5))

ax.plot(init_traj[:,0], init_traj[:,1], 'r--', label='Initial Guess (Colliding)')
ax.plot(planned_traj[:,0], planned_traj[:,1], 'b-', label='iLQR Plan (IFT Aware)')
ax.plot(actual_states[:,0], actual_states[:,1], 'g-', alpha=0.6, label='True Performance (Noisy)')
ax.scatter(goal_pose[0], goal_pose[1], marker='X', color='green', s=100, label='Goal')

ax.legend(); plt.title("Physics-Based Trajectory Optimization"); plt.show()