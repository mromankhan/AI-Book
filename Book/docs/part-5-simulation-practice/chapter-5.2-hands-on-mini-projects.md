---
sidebar_position: 2
---

# Chapter 5.2: Hands-on Mini Projects

## Overview

In this chapter, you will learn:
- How to implement a line-following robot simulation
- How to create an obstacle avoidance system
- How to develop simple humanoid walking logic
- How to apply the concepts learned in this book to practical projects

These mini projects serve as practical applications of the Physical AI concepts covered throughout this book, allowing you to implement and test the principles in simulated environments.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Implement a complete line-following robot with sensor feedback and control
2. Build an obstacle avoidance system for navigation
3. Develop basic humanoid walking patterns using balance control
4. Apply the principles of simulation-to-reality transfer in practical scenarios

### Why This Matters

Mini projects provide practical experience with the core concepts of Physical AI, allowing you to integrate perception, control, and decision-making components. These hands-on exercises bridge the gap between theoretical knowledge and practical implementation, preparing you for more complex Physical AI challenges.

## Core Concepts

### Project 1: Line Follower Robot

A classic robotics project that integrates:
- **Perception**: Reading sensor data from a virtual line-following sensor array
- **Control**: Implementing a feedback control system to track the line
- **Navigation**: Following a predetermined path through an environment

For beginners: A line follower robot is like a person following a marked path on the ground, using their eyes to stay on the path.

For intermediate learners: The line follower is a practical example of feedback control with sensor fusion, demonstrating PID control principles.

### Project 2: Obstacle Avoidance

An essential navigation skill that combines:
- **Perception**: Detecting obstacles in the environment
- **Decision Making**: Choosing appropriate actions based on sensor inputs
- **Control**: Adjusting motion to avoid collisions while maintaining goal progress

For beginners: This is like walking through a crowded room, watching for obstacles and adjusting your path to avoid bumping into people.

For intermediate learners: This demonstrates reactive and deliberative planning, sensor fusion, and control theory applied to navigation.

### Project 3: Simple Humanoid Walking

A complex integration project involving:
- **Balance Control**: Maintaining stability while moving
- **Locomotion**: Generating periodic gait patterns
- **Physics Simulation**: Managing the complex dynamics of bipedal movement

For beginners: This is like learning to walk again using mechanical legs, with careful attention to balance.

For intermediate learners: This demonstrates advanced control techniques including ZMP control, inverse kinematics, and dynamic stability.

### Physical AI Integration

These projects demonstrate the integration of multiple Physical AI disciplines:
- **Perception**: Sensory data interpretation
- **Control**: Motion generation and regulation
- **Planning**: Path and trajectory generation
- **Learning**: Adapting to new situations (in advanced implementations)

For beginners: Physical AI brings together all the components of a robot to act intelligently in the physical world.

For intermediate learners: These systems require careful integration of multiple subsystems with appropriate information flow and control architecture.

## Hands-on Section

### Project 1: Line Follower Robot

Let's create a complete line-following robot simulation:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

class LineFollowerRobot:
    """A simulated line-following robot"""
    def __init__(self, start_pos=(0, 0), start_angle=0):
        # Robot state: [x, y, theta, v]
        self.state = np.array([start_pos[0], start_pos[1], start_angle, 0.0])  # [x, y, theta, v]
        self.length = 0.2  # Robot length (m)
        self.width = 0.15  # Robot width (m)
        self.wheel_base = 0.15  # Distance between front and rear axles
        self.max_speed = 0.5  # m/s
        self.max_steering = np.pi/3  # Max steering angle (60 degrees)
        self.sensor_array_offset = 0.1  # Position of sensors from robot center (m)
        self.sensor_count = 5  # Number of sensors in the array
        self.sensor_spacing = 0.03  # Spacing between sensors (m)
        self.dt = 0.02  # Time step (s)
        
    def sense_line(self, line_func, noise_std=0.01):
        """Sense the line under the robot's sensor array"""
        x, y, theta, v = self.state
        
        # Calculate sensor positions
        sensor_positions = []
        for i in range(self.sensor_count):
            # Sensors are arranged perpendicular to the robot's heading
            offset = (i - (self.sensor_count - 1) / 2) * self.sensor_spacing
            sens_x = x + (self.sensor_array_offset * np.cos(theta) - offset * np.sin(theta))
            sens_y = y + (self.sensor_array_offset * np.sin(theta) + offset * np.cos(theta))
            sensor_positions.append((sens_x, sens_y))
        
        # Sample the line function at each sensor position
        sensor_values = []
        for sx, sy in sensor_positions:
            # Calculate distance to the line (simplified to y-axis for demonstration)
            expected_line_y = line_func(sx)
            distance_to_line = sy - expected_line_y
            # Convert to sensor reading (0=off line, 1=on line)
            # Use Gaussian to simulate sensor width
            reading = np.exp(-(distance_to_line**2) / (2 * 0.01**2))
            # Add noise
            reading += np.random.normal(0, noise_std)
            reading = np.clip(reading, 0, 1)
            sensor_values.append(reading)
        
        return np.array(sensor_values)
    
    def compute_steering(self, sensor_readings, kp=2.0, kd=0.1):
        """Compute steering command from sensor readings"""
        # Calculate weighted average of sensor positions based on readings
        # This gives us the "centroid" of the line under the sensors
        positions = np.arange(len(sensor_readings)) - (len(sensor_readings) - 1) / 2
        total_weight = np.sum(sensor_readings)
        
        if total_weight > 0.1:  # If we detect the line
            centroid = np.sum(positions * sensor_readings) / total_weight
        else:  # If no line detected, maintain current heading
            centroid = 0
        
        # Simple PD controller
        error = -centroid  # Negative because sensors are reversed
        prev_error = getattr(self, 'prev_error', 0)
        error_rate = (error - prev_error) / self.dt if self.dt > 0 else 0
        
        steering = kp * error + kd * error_rate
        
        # Store for next iteration
        self.prev_error = error
        
        # Limit steering
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        return steering
    
    def update(self, steering_cmd, speed_cmd):
        """Update robot state based on steering and speed commands"""
        x, y, theta, v = self.state
        
        # Limit commands
        steering = np.clip(steering_cmd, -self.max_steering, self.max_steering)
        speed = np.clip(speed_cmd, -self.max_speed, self.max_speed)
        
        # Bicycle model for robot kinematics
        beta = np.arctan(0.5 * np.tan(steering))  # Slip angle
        
        # State derivatives
        dx = v * np.cos(theta + beta)
        dy = v * np.sin(theta + beta)
        dtheta = v * np.cos(beta) * np.tan(steering) / self.wheel_base
        dv = (speed - v) / 0.1  # Simple acceleration model with 0.1s time constant
        
        # Update state
        self.state[0] += dx * self.dt
        self.state[1] += dy * self.dt
        self.state[2] += dtheta * self.dt
        self.state[3] += dv * self.dt
        
        # Normalize angle
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi
        
        return self.state.copy()
    
    def get_position(self):
        """Get robot position [x, y]"""
        return self.state[:2]
    
    def get_heading(self):
        """Get robot heading (theta)"""
        return self.state[2]

# Define a line path (curved for interest)
def curved_line_path(x):
    """Curved line path for robot to follow"""
    return 0.5 * np.sin(2 * x / 3.0)

# Create robot and run simulation
robot = LineFollowerRobot(start_pos=(0, 0.5), start_angle=0)

# Define the path (for visualization)
path_x = np.linspace(0, 10, 1000)
path_y = curved_line_path(path_x)

T = 20.0  # Simulation time
t_eval = np.arange(0, T, robot.dt)

# Store robot trajectory
trajectory_x = [robot.state[0]]
trajectory_y = [robot.state[1]]
headings = [robot.state[2]]
sensor_logs = []
time_points = [0]

# Run the simulation
for i, t in enumerate(t_eval[1:], 1):
    # Sense the line
    sensors = robot.sense_line(curved_line_path)
    
    # Compute control based on sensor readings
    steering = robot.compute_steering(sensors, kp=2.0, kd=0.1)
    
    # Set speed (slow down in curves)
    curvature = np.abs(2.0/3.0 * np.cos(2 * robot.state[0] / 3.0))
    base_speed = 0.3
    speed = base_speed / (1 + 5*curvature)  # Slow down in sharp curves
    
    # Update robot
    state = robot.update(steering, speed)
    
    # Log data
    trajectory_x.append(state[0])
    trajectory_y.append(state[1])
    headings.append(state[2])
    sensor_logs.append(sensors.copy())
    time_points.append(t)

# Visualize the results
plt.figure(figsize=(18, 12))

# Plot the robot path
plt.subplot(2, 2, 1)
plt.plot(path_x, path_y, 'r-', linewidth=3, label='Reference Line', alpha=0.7)
plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Robot Path')
plt.scatter(trajectory_x[0], trajectory_y[0], c='green', s=100, label='Start', zorder=5)
plt.scatter(trajectory_x[-1], trajectory_y[-1], c='red', s=100, label='End', zorder=5)

# Show robot orientations at intervals
for i in range(0, len(trajectory_x), 20):
    x, y, theta = trajectory_x[i], trajectory_y[i], headings[i]
    dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
    plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05, fc='blue', ec='blue')

plt.title('Line Follower Robot Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot sensor readings over time
plt.subplot(2, 2, 2)
sensor_logs = np.array(sensor_logs)
for i in range(robot.sensor_count):
    plt.subplot(2, 2, 2)
    plt.plot(time_points[1:], sensor_logs[:, i], label=f'Sensor {i+1}', linewidth=2)
plt.title('Sensor Readings Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot cross-track error
plt.subplot(2, 2, 3)
y_expected = curved_line_path(np.array(trajectory_x[1:]))
cross_track_error = np.array(trajectory_y[1:]) - y_expected
plt.plot(time_points[1:], cross_track_error, 'r-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.title('Cross-Track Error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.grid(True, alpha=0.3)

# Plot robot speed
plt.subplot(2, 2, 4)
speeds = [robot.state[3] for _ in range(len(time_points))]  # Actual speeds
plt.plot(time_points, speeds, 'g-', linewidth=2, label='Actual Speed')
plt.title('Robot Speed Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate performance metrics
avg_error = np.mean(np.abs(cross_track_error))
max_error = np.max(np.abs(cross_track_error))
total_distance = np.sum(np.sqrt(np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2))

print(f"\nLine Follower Performance Metrics:")
print(f"Average cross-track error: {avg_error:.3f} m")
print(f"Maximum cross-track error: {max_error:.3f} m")
print(f"Total distance traveled: {total_distance:.2f} m")
print(f"Final position: ({trajectory_x[-1]:.3f}, {trajectory_y[-1]:.3f})")
```

### Project 2: Obstacle Avoidance System

Now let's create an obstacle avoidance simulation:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import cdist

class ObstacleAvoidanceRobot:
    """A simulated robot with obstacle avoidance capabilities"""
    def __init__(self, start_pos=(0, 0), goal_pos=(10, 10)):
        # Robot state: [x, y, theta, v]
        self.state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])  # [x, y, theta, v]
        self.goal = np.array(goal_pos)
        self.radius = 0.2  # Robot radius (m)
        self.max_speed = 0.8  # m/s
        self.max_omega = np.pi/2  # Max angular velocity (rad/s)
        self.laser_range = 3.0  # Laser scanner range (m)
        self.laser_resolution = np.pi / 180  # 1 degree resolution
        self.safe_distance = 0.3  # Safe distance from obstacles (m)
        self.dt = 0.05  # Time step (s)
        
    def sense_environment(self, obstacles, noise_std=0.02):
        """Simulate laser range finder sensor"""
        x, y, theta, v = self.state
        
        # Generate laser angles (front + 180 degrees field of view)
        angles = np.arange(-np.pi/2, np.pi/2 + self.laser_resolution, self.laser_resolution)
        angles = angles + theta  # Account for robot orientation
        
        ranges = []
        for angle in angles:
            # Ray from robot in direction of sensor
            ray_cos = np.cos(angle)
            ray_sin = np.sin(angle)
            
            # Check distance to each obstacle
            min_distance = self.laser_range
            
            for obs_x, obs_y, obs_radius in obstacles:
                # Vector from robot to obstacle center
                vec_to_center = np.array([obs_x - x, obs_y - y])
                
                # Project this vector onto the ray direction
                proj_len = vec_to_center[0] * ray_cos + vec_to_center[1] * ray_sin
                
                # Closest point on ray to obstacle center
                closest_on_ray = np.array([x + proj_len * ray_cos, y + proj_len * ray_sin])
                
                # Distance from closest point to obstacle center
                dist_to_center = np.linalg.norm(closest_on_ray - np.array([obs_x, obs_y]))
                
                # If ray intersects obstacle
                if dist_to_center < obs_radius and proj_len > 0:
                    # Calculate entry and exit points of ray-ellipse intersection
                    # Simplified to ray-circle intersection
                    discriminant = obs_radius**2 - dist_to_center**2
                    
                    if discriminant >= 0:
                        d_intersect = np.sqrt(discriminant)
                        entry_dist = proj_len - d_intersect
                        
                        if entry_dist > 0:  # Entry point is in front
                            min_distance = min(min_distance, entry_dist)
            
            # Add some noise
            min_distance += np.random.normal(0, noise_std)
            min_distance = np.clip(min_distance, 0.01, self.laser_range)
            ranges.append(min_distance)
        
        return np.array(ranges), angles
    
    def potential_field_navigation(self, obstacles, sensor_ranges, sensor_angles):
        """Use potential field method for navigation"""
        x, y, theta, v = self.state
        
        # Attractive force toward goal
        goal_vec = self.goal - np.array([x, y])
        goal_dist = np.linalg.norm(goal_vec)
        
        # Define attractive potential parameters
        attractive_gain = 0.5
        attractive_force = attractive_gain * goal_vec if goal_dist > 0 else np.array([0, 0])
        
        # Repulsive forces from obstacles
        repulsive_force = np.array([0.0, 0.0])
        
        for i, (range_val, angle) in enumerate(zip(sensor_ranges, sensor_angles)):
            if range_val < self.safe_distance * 2:  # Only consider nearby obstacles
                # Calculate direction of this sensor
                direction = np.array([np.cos(angle), np.sin(angle)])
                
                # Repulsive force magnitude (stronger when closer)
                repulsion_strength = 1.0 / max(range_val, 0.05)  # Avoid division by zero
                repulsion_strength *= (1.0 - min(range_val, self.safe_distance) / self.safe_distance)
                
                # Add repulsive force in opposite direction of obstacle
                repulsive_force -= repulsion_strength * direction * 0.5
        
        # Combine forces
        total_force = attractive_force + repulsive_force
        
        # Calculate desired heading
        if np.linalg.norm(total_force) > 0.01:
            desired_theta = np.arctan2(total_force[1], total_force[0])
        else:
            desired_theta = theta  # Maintain current heading if no clear direction
        
        # Calculate angular velocity needed to turn toward desired heading
        theta_error = desired_theta - theta
        # Normalize angle error to [-pi, pi]
        while theta_error > np.pi:
            theta_error -= 2 * np.pi
        while theta_error < -np.pi:
            theta_error += 2 * np.pi
        
        # PID controller for orientation
        k_p = 2.0
        k_d = 0.1
        prev_theta_error = getattr(self, 'prev_theta_error', 0)
        theta_error_rate = (theta_error - prev_theta_error) / self.dt if self.dt > 0 else 0
        
        omega_cmd = k_p * theta_error + k_d * theta_error_rate
        self.prev_theta_error = theta_error
        
        # Limit angular velocity
        omega_cmd = np.clip(omega_cmd, -self.max_omega, self.max_omega)
        
        # Adjust speed based on obstacle proximity
        min_range = min(sensor_ranges) if len(sensor_ranges) > 0 else self.laser_range
        speed_factor = min_range / (2 * self.safe_distance)
        speed_factor = np.clip(speed_factor, 0.2, 1.0)  # Slow down near obstacles
        speed_cmd = self.max_speed * speed_factor
        
        return speed_cmd, omega_cmd
    
    def update(self, v_cmd, omega_cmd):
        """Update robot state based on velocity commands"""
        x, y, theta, v = self.state
        
        # Limit commands
        v_cmd = np.clip(v_cmd, -self.max_speed, self.max_speed)
        omega_cmd = np.clip(omega_cmd, -self.max_omega, self.max_omega)
        
        # Update state using bicycle model
        self.state[0] += v_cmd * np.cos(theta) * self.dt
        self.state[1] += v_cmd * np.sin(theta) * self.dt
        self.state[2] += omega_cmd * self.dt
        self.state[3] = v_cmd  # Update current speed
        
        # Normalize angle
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi
        
        return self.state.copy()

# Define environment with obstacles
obstacles = [
    (3, 3, 0.5),    # (x, y, radius)
    (5, 7, 0.4),
    (7, 4, 0.6),
    (2, 8, 0.3),
    (8, 8, 0.4),
]

# Create robot
robot = ObstacleAvoidanceRobot(start_pos=(1, 1), goal_pos=(9, 9))

# Run simulation
T = 30.0  # Simulation time
t_eval = np.arange(0, T, robot.dt)

# Store robot trajectory
trajectory_x = [robot.state[0]]
trajectory_y = [robot.state[1]]
headings = [robot.state[2]]
avoidance_active = []  # Track when obstacle avoidance is active

# Run the simulation
for i, t in enumerate(t_eval[1:], 1):
    # Sense environment
    sensor_ranges, sensor_angles = robot.sense_environment(obstacles)
    
    # Check if obstacles are detected in front
    front_ranges = sensor_ranges[len(sensor_ranges)//3:2*len(sensor_ranges)//3]
    min_front_range = min(front_ranges) if len(front_ranges) > 0 else robot.laser_range
    
    # Apply potential field navigation
    speed_cmd, omega_cmd = robot.potential_field_navigation(obstacles, sensor_ranges, sensor_angles)
    
    # Check if obstacle avoidance is active
    obstacle_nearby = min_front_range < robot.safe_distance * 1.5
    avoidance_active.append(obstacle_nearby)
    
    # Update robot
    state = robot.update(speed_cmd, omega_cmd)
    
    # Log data
    trajectory_x.append(state[0])
    trajectory_y.append(state[1])
    headings.append(state[2])

# Calculate distance to goal over time
distances_to_goal = [np.linalg.norm(robot.goal - np.array([x, y])) 
                     for x, y in zip(trajectory_x, trajectory_y)]

print(f"\nObstacle Avoidance Simulation Results:")

# Visualize the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Plot the robot path with obstacles
ax1.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Robot Path')
ax1.scatter([robot.goal[0]], [robot.goal[1]], c='red', s=200, label='Goal', zorder=5)
ax1.scatter([1], [1], c='green', s=200, label='Start', zorder=5)

# Draw obstacles
for obs_x, obs_y, obs_radius in obstacles:
    circle = patches.Circle((obs_x, obs_y), obs_radius, color='red', alpha=0.3)
    ax1.add_patch(circle)

# Show robot orientations at intervals
for i in range(0, len(trajectory_x), 20):
    x, y, theta = trajectory_x[i], trajectory_y[i], headings[i]
    dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
    ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

ax1.set_title('Obstacle Avoidance Path')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot distance to goal over time
ax2.plot(t_eval, distances_to_goal, 'g-', linewidth=2)
ax2.axhline(y=1.0, color='r', linestyle='--', label='Goal Reached (1m)', alpha=0.7)
ax2.set_title('Distance to Goal Over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Goal (m)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot robot speed over time
speeds = [robot.state[3] for _ in range(len(t_eval))]
ax3.plot(t_eval, speeds, 'orange', linewidth=2, label='Robot Speed')
ax3.set_title('Robot Speed Over Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Speed (m/s)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot obstacle avoidance activity
ax4.plot(t_eval[1:], avoidance_active, 'r-', linewidth=2)
ax4.set_title('Obstacle Avoidance Activity')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Avoidance Active')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['No Obstacles', 'Avoiding'])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate performance metrics
goal_reached = distances_to_goal[-1] < 1.0  # Within 1m of goal
path_efficiency = np.sum(np.sqrt(np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2)) / \
                  np.linalg.norm(robot.goal - np.array([1, 1]))  # Actual distance / straight-line distance
avg_speed = np.mean(speeds)
obstacle_avoidance_time = sum(avoidance_active) / len(avoidance_active)  # Fraction of time avoiding obstacles

print(f"Goal reached: {'Yes' if goal_reached else 'No'}")
print(f"Path efficiency: {path_efficiency:.3f} (lower is better)")
print(f"Average speed: {avg_speed:.3f} m/s")
print(f"Fraction of time in obstacle avoidance: {obstacle_avoidance_time:.2f}")
print(f"Final distance to goal: {distances_to_goal[-1]:.3f} m")
```

### Project 3: Simple Humanoid Walking

Finally, let's create a simple humanoid walking simulation:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SimpleHumanoidWalker:
    """A simplified model of a bipedal walker"""
    def __init__(self):
        # Robot parameters
        self.height = 0.9  # Height of CoM (m)
        self.mass = 50.0   # Mass (kg)
        self.gravity = 9.81  # Gravity (m/s^2)
        self.leg_length = 0.8  # Leg length (m)
        
        # State: [x_com, z_com, theta_left, theta_right, xdot_com, zdot_com, thetadot_left, thetadot_right]
        # theta: angle from vertical of each leg
        self.state = np.array([0.0, self.height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Gait parameters
        self.stride_frequency = 1.0  # Steps per second
        self.stride_length = 0.6     # Desired step length (m)
        self.step_height = 0.1       # Swing foot clearance (m)
        self.phase = 0.0             # Gait phase [0, 1]
        
        self.dt = 0.01  # Time step
        
    def get_leg_positions(self, state):
        """Calculate foot positions from leg angles"""
        x_com, z_com, theta_l, theta_r, _, _, _, _ = state
        
        # Calculate foot positions relative to CoM
        left_foot_x = x_com + self.leg_length * np.sin(theta_l)
        left_foot_z = z_com - self.leg_length * np.cos(theta_l)
        
        right_foot_x = x_com + self.leg_length * np.sin(theta_r)
        right_foot_z = z_com - self.leg_length * np.cos(theta_r)
        
        return (left_foot_x, left_foot_z), (right_foot_x, right_foot_z)
    
    def update_gait_phase(self):
        """Update the gait phase"""
        self.phase = (self.phase + self.stride_frequency * self.dt) % 1.0
        return self.phase
    
    def balance_controller(self, state):
        """Simple balance controller to regulate CoM position"""
        x_com, z_com, theta_l, theta_r, xdot_com, zdot_com, thetadot_l, thetadot_r = state
        
        # Simple control to keep CoM at desired height and near center
        desired_x = 0.0 if self.phase < 0.5 else self.stride_length / 2  # Move CoM toward support foot
        desired_z = self.height
        
        # Calculate errors
        x_error = desired_x - x_com
        z_error = desired_z - z_com
        
        # Simple PD control
        kp_x, kd_x = 10.0, 2.0
        kp_z, kd_z = 100.0, 20.0  # Higher gain for vertical stability
        
        # Calculate desired accelerations
        x_ddot_cmd = kp_x * x_error - kd_x * xdot_com
        z_ddot_cmd = kp_z * z_error - kd_z * zdot_com  # Only gravity correction for z
        
        return x_ddot_cmd, z_ddot_cmd
    
    def leg_controller(self, state, phase):
        """Control leg angles to achieve desired gait"""
        x_com, z_com, theta_l, theta_r, xdot_com, zdot_com, thetadot_l, thetadot_r = state
        
        # Determine support and swing legs based on phase
        if phase < 0.5:
            # Left leg is stance, right leg is swing
            support_leg = 'left'
            swing_leg = 'right'
            swing_phase = phase * 2  # Map 0.0-0.5 to 0.0-1.0
        else:
            # Right leg is stance, left leg is swing
            support_leg = 'right'
            swing_leg = 'left'
            swing_phase = (phase - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
        
        # Calculate forces needed to achieve desired CoM acceleration
        x_ddot_cmd, z_ddot_cmd = self.balance_controller(state)
        
        # Simple inverse dynamics to determine required torques
        # For swing leg, we want to follow a trajectory
        if swing_leg == 'left':
            # Desired trajectory for left leg
            desired_theta_l = 0.1 * np.sin(2 * np.pi * swing_phase)  # Swing forward
            desired_thetadot_l = 0.0
            
            # For stance leg, keep more vertical to support body
            desired_theta_r = 0.0  # Keep close to vertical
            desired_thetadot_r = 0.0
        else:  # Swing leg is right
            desired_theta_r = 0.1 * np.sin(2 * np.pi * swing_phase)  # Swing forward
            desired_thetadot_r = 0.0
            
            # For stance leg, keep more vertical to support body
            desired_theta_l = 0.0  # Keep close to vertical
            desired_thetadot_l = 0.0
        
        # Calculate torques needed to achieve desired joint kinematics
        kp_theta, kd_theta = 50.0, 5.0
        
        tau_l = kp_theta * (desired_theta_l - theta_l) + kd_theta * (desired_thetadot_l - thetadot_l)
        tau_r = kp_theta * (desired_theta_r - theta_r) + kd_theta * (desired_thetadot_r - thetadot_r)
        
        return tau_l, tau_r
    
    def dynamics(self, t, state):
        """Equations of motion for the humanoid model"""
        # State vector unpacking
        x_com, z_com, theta_l, theta_r, xdot_com, zdot_com, thetadot_l, thetadot_r = state
        
        # Compute control inputs
        phase = self.phase
        tau_l, tau_r = self.leg_controller(state, phase)
        
        # Simple inverted pendulum dynamics with leg control
        # This is a simplified model - real humanoid dynamics are much more complex
        
        # Calculate accelerations (simplified inverted pendulum model)
        x_ddot = 0.0  # This will be computed based on balance control
        z_ddot = -self.gravity  # Gravity acts downward
        
        # Leg dynamics (simplified - real models would have complex coupled dynamics)
        theta_l_ddot = tau_l / (self.mass * self.leg_length**2)  # Torque = I * alpha
        theta_r_ddot = tau_r / (self.mass * self.leg_length**2)
        
        # Update state derivatives
        derivatives = [
            xdot_com,           # dx/dt
            zdot_com,           # dz/dt
            thetadot_l,         # dtheta_l/dt
            thetadot_r,         # dtheta_r/dt
            x_ddot,             # dxdot/dt
            z_ddot,             # dzdot/dt
            theta_l_ddot,       # dthetadot_l/dt
            theta_r_ddot        # dthetadot_r/dt
        ]
        
        return derivatives
    
    def step(self):
        """Step the simulation forward by dt"""
        # Update gait phase
        phase = self.update_gait_phase()
        
        # Integrate the dynamics
        t_span = [0, self.dt]
        sol = solve_ivp(self.dynamics, t_span, self.state, method='RK45')
        
        # Update state with solution
        self.state = sol.y[:, -1]
        
        # Apply constraints (preventing legs from going through ground)
        x_com, z_com, theta_l, theta_r, xdot_com, zdot_com, thetadot_l, thetadot_r = self.state
        
        # Calculate foot positions
        (lf_x, lf_z), (rf_x, rf_z) = self.get_leg_positions(self.state)
        
        # If foot goes below ground, adjust state to keep it at ground level
        if lf_z < 0:
            # Adjust CoM height to keep left foot at ground
            z_correction = -lf_z
            self.state[1] += z_correction  # Adjust CoM z position
            self.state[5] = 0  # Zero z velocity
        if rf_z < 0:
            # Adjust CoM height to keep right foot at ground
            z_correction = -rf_z
            self.state[1] += z_correction  # Adjust CoM z position
            self.state[5] = 0  # Zero z velocity
        
        # Return current state
        return self.state.copy()

# Run the walking simulation
walker = SimpleHumanoidWalker()
T = 10.0  # Simulation time
t_eval = np.arange(0, T, walker.dt)

# Store data
com_positions = []
leg_angles = []
gait_phases = []
foot_positions = []

for t in t_eval:
    state = walker.step()
    
    # Log data
    com_pos = [state[0], state[1]]  # x_com, z_com
    leg_ang = [state[2], state[3]]  # theta_left, theta_right
    phase = walker.phase
    
    com_positions.append(com_pos.copy())
    leg_angles.append(leg_ang.copy())
    gait_phases.append(phase)
    
    # Calculate foot positions from current state
    left_foot, right_foot = walker.get_leg_positions(state)
    foot_positions.append([left_foot, right_foot])

# Convert to numpy arrays
com_positions = np.array(com_positions)
leg_angles = np.array(leg_angles)
gait_phases = np.array(gait_phases)
foot_positions = np.array(foot_positions)

# Extract separate components
com_x = com_positions[:, 0]
com_z = com_positions[:, 1]
left_leg_angle = leg_angles[:, 0]
right_leg_angle = leg_angles[:, 1]
left_foot_x = foot_positions[:, 0, 0]
left_foot_z = foot_positions[:, 0, 1]
right_foot_x = foot_positions[:, 1, 0]
right_foot_z = foot_positions[:, 1, 1]

# Calculate walking metrics
step_times = [i*walker.dt for i, phase in enumerate(gait_phases) if abs(phase) < 0.01 or abs(phase-0.5) < 0.01]
stride_lengths = np.diff(com_x[::int(1/(walker.dt*walker.stride_frequency))]) if len(com_x) > int(1/(walker.dt*walker.stride_frequency)) else [walker.stride_length]*10

# Plot the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Plot CoM trajectory
ax1.plot(com_x, com_z, 'b-', linewidth=2, label='CoM Path')
ax1.plot(left_foot_x, left_foot_z, 'r--', linewidth=1, label='Left Foot Path', alpha=0.7)
ax1.plot(right_foot_x, right_foot_z, 'g--', linewidth=1, label='Right Foot Path', alpha=0.7)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=1, label='Ground')
ax1.plot(com_x[0], com_z[0], 'go', markersize=10, label='Start', zorder=5)
ax1.plot(com_x[-1], com_z[-1], 'ro', markersize=10, label='End', zorder=5)
ax1.set_title('Humanoid Walking Trajectory')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Z Position (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot leg angles over time
ax2.plot(t_eval, left_leg_angle, label='Left Leg Angle', linewidth=2)
ax2.plot(t_eval, right_leg_angle, label='Right Leg Angle', linewidth=2)
ax2.set_title('Leg Angles Over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot gait phase over time
ax3.plot(t_eval, gait_phases, 'purple', linewidth=2)
ax3.set_title('Gait Phase Over Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Phase (0-1)')
ax3.grid(True, alpha=0.3)

# Plot step-by-step position
ax4.plot(t_eval, com_x, 'b-', linewidth=2, label='CoM X Position')
ax4.plot(t_eval, com_z, 'r-', linewidth=2, label='CoM Z Position')
ax4.set_title('CoM Position Components Over Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Position (m)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate performance metrics
avg_forward_speed = (com_x[-1] - com_x[0]) / T
avg_height_variation = np.std(com_z)
avg_stride_length = np.mean(stride_lengths) if stride_lengths else 0

print(f"\nHumanoid Walking Simulation Results:")
print(f"Average forward speed: {avg_forward_speed:.3f} m/s")
print(f"Average CoM height variation: {avg_height_variation:.3f} m")
print(f"Average stride length: {avg_stride_length:.3f} m")
print(f"Distance traveled: {com_x[-1] - com_x[0]:.2f} m")
print(f"Final CoM position: ({com_x[-1]:.3f}, {com_z[-1]:.3f}) m")
```

## Real-World Mapping

### Mini Projects in Practice

The mini projects implemented here map to real-world robotics applications:

- **Line Follower**: Used in automated guided vehicles (AGVs) in warehouses and factories
- **Obstacle Avoidance**: Core technology for autonomous vehicles, drones, and service robots
- **Humanoid Walking**: Foundation for bipedal robots used in research and specialized applications

### Simulation vs. Reality Considerations

| Project | Simulation Strengths | Real-World Challenges |
|---------|---------------------|----------------------|
| **Line Follower** | Perfect sensor modeling | Real sensor noise & variability |
| | Known, consistent environment | Changing lighting conditions |
| | No mechanical wear | Wheel slip, motor inconsistencies |
| **Obstacle Avoidance** | Accurate environment modeling | Sensor limitations in complex scenes |
| | No safety risks | Safety requirements for real robots |
| | Fast iteration | Real-time computational constraints |
| **Humanoid Walking** | Simplified physics | Complex multi-body dynamics |
| | Known robot parameters | Parameter uncertainty |
| | No falls | Need to prevent actual falls |

### Development Best Practices

For successful project implementation:

1. **Start Simple**: Begin with basic models and gradually add complexity
2. **Validate Early**: Test individual components before integrating
3. **Iterate Frequently**: Regular testing and adjustment
4. **Consider Transfer**: Design with sim-to-real transfer in mind

### Advanced Extensions

To further develop these projects:

1. **Line Follower**:
   - Add more complex path shapes
   - Implement machine learning-based control
   - Add multiple robot coordination

2. **Obstacle Avoidance**:
   - Use more sophisticated path planning
   - Add dynamic obstacles
   - Implement formation control

3. **Humanoid Walking**:
   - Add upper body control
   - Implement more advanced balance strategies
   - Add stair climbing or stepping over obstacles

## Exercises

### Beginner Tasks
1. Run the line follower simulation and experiment with different PID gains
2. Try the obstacle avoidance simulation with different obstacle layouts
3. Run the walking simulation and observe how the robot balances
4. Modify the sensor noise in any of the simulations to see its effect

### Stretch Challenges
1. Combine the line follower and obstacle avoidance into a single navigation system
2. Implement a learning algorithm to optimize the PID parameters for the line follower
3. Create a humanoid walker that can climb stairs

## Summary

This chapter provided hands-on implementation of three classic robotics projects: line following, obstacle avoidance, and humanoid walking. These mini projects demonstrate the integration of perception, control, and decision-making components essential to Physical AI.

Each project builds on the concepts covered throughout this book, showing how to combine sensors, actuators, and control algorithms to achieve complex behaviors in simulation. These foundational projects prepare you for more complex Physical AI challenges in real-world applications.

In the next part of this book, we'll explore system integration and how all these components work together in complete Physical AI systems.