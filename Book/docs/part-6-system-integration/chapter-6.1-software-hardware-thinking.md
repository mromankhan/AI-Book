---
sidebar_position: 1
---

# Chapter 6.1: Software + Hardware Thinking

## Overview

In this chapter, you will learn:
- How to think about systems holistically considering both software and hardware
- How to understand failure modes in Physical AI systems
- How to debug physical systems effectively
- How to design software that properly accounts for physical constraints

Physical AI systems require a combined approach that considers both software algorithms and physical hardware constraints as inseparable components of the system.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Think in terms of software-hardware integration rather than separate components
2. Identify and analyze system failure modes in Physical AI
3. Apply systematic debugging techniques to physical systems
4. Design software that properly accounts for hardware limitations and constraints

### Why This Matters

Physical AI systems are fundamentally different from purely software systems because the software and hardware must work together as one inseparable unit. Understanding this integration is critical for creating robust, reliable systems that can operate safely in the physical world.

## Core Concepts

### End-to-End System View

Physical AI systems require considering the complete chain from sensor input to actuator output:

1. **Perception**: Sensors gather information about the world
2. **Processing**: Algorithms interpret sensory data
3. **Decision Making**: Higher-level algorithms determine actions
4. **Control**: Systems convert high-level goals to motor commands
5. **Actuation**: Physical systems execute actions
6. **Feedback**: Results are sensed and the cycle repeats

For beginners: This is like thinking of a robot as a complete living system, where all parts must work together - like how your eyes, brain, and muscles work together to catch a ball.

For intermediate learners: This requires understanding the latency, bandwidth, and accuracy constraints of each system element and how they compound.

### Software-Hardware Integration

Key aspects of tight integration include:

1. **Timing Constraints**: Hardware has physical response times that software must respect
2. **Resource Limitations**: Power, processing, and memory constraints affect algorithm selection
3. **Physical Constraints**: Hardware limitations directly affect software capabilities
4. **Sensor Accuracy**: Noise and uncertainty in sensors affects algorithm performance
5. **Actuator Capabilities**: What the hardware can do constrains what the software can achieve

For beginners: It's like having a recipe that must consider not only what you want to cook but also what ingredients you have, how long things take to cook, and what equipment you have available.

For intermediate learners: This requires understanding system-level trade-offs between different design choices and their implications across the entire system.

### Failure Mode Analysis

Systems can fail in various ways:

1. **Component Level**: Individual sensors, actuators, or processors fail
2. **Integration Level**: Components don't work together as expected
3. **System Level**: The system doesn't achieve its intended function
4. **Environmental Level**: System fails in unexpected environmental conditions
5. **Interaction Level**: System behaves safely but unexpectedly with humans/environment

For beginners: Imagine a robot that stops working because its wheel gets stuck or a software bug causes it to go in circles.

For intermediate learners: Failure mode analysis requires understanding complex interactions and often requires formal methods or extensive testing.

### Debugging Physical Systems

Debugging combines physical and software debugging:

1. **Isolation**: Separating software from hardware problems
2. **Instrumentation**: Adding sensors and logging to understand system state
3. **Repeatability**: Creating reproducible test conditions
4. **Safety**: Ensuring debugging doesn't cause harm during physical tests
5. **Logging**: Recording complete system states for offline analysis

For beginners: Like troubleshooting a broken toy, you need to figure out if it's the battery (hardware) or the switch mechanism (software).

For intermediate learners: This requires understanding the interplay between software algorithms, system timing, sensor feedback, and physical hardware behavior.

## Hands-on Section

### System Integration Simulation

Let's create a simulation that demonstrates the integration of all components:

```python
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue

class PhysicalAISystem:
    """A complete Physical AI system simulation demonstrating software-hardware integration"""
    def __init__(self):
        # System state
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.orientation = 0.0  # Robot heading in radians
        self.angular_velocity = 0.0
        
        # Hardware characteristics
        self.mass = 5.0  # kg
        self.max_force = 10.0  # N
        self.max_torque = 5.0  # N⋅m
        self.sensor_range = 5.0  # m
        self.sensor_noise_std = 0.02  # Sensor noise in meters
        self.actuator_delay = 0.02  # s (actuator response delay)
        self.control_loop_period = 0.05  # s
        
        # System components simulation
        self.perception_module = PerceptionModule()
        self.control_module = ControlModule()
        self.motion_planner = MotionPlanner()
        self.safety_system = SafetySystem()
        
        # Logging
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.command_history = []
        self.error_history = []
        
        # Threading
        self.stop_flag = False
        self.system_lock = threading.Lock()
        
    def sense_environment(self, environment_state):
        """Simulate sensor data acquisition"""
        # Simulate range sensors around the robot
        sensor_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16 sensors evenly spaced
        sensor_data = []
        
        for angle in sensor_angles:
            # Add noise to sensor readings
            noise = np.random.normal(0, self.sensor_noise_std)
            # Simulate sensor reading (in a simple environment)
            distance = self.sensor_range  # Default to max range
            # Add simulated obstacles for testing
            if abs(angle) < np.pi/4:  # Front-facing sensors
                distance = 2.0 + noise  # Obstacle in front
            elif abs(angle - np.pi/2) < np.pi/6:  # Right side sensors
                distance = 1.5 + noise  # Obstacle to the right
            else:
                distance = min(self.sensor_range, 3.0 + noise)  # Free space elsewhere
                
            sensor_data.append(max(0.05, distance))  # Min range 5cm, max range sensor_range
        
        return np.array(sensor_data)
    
    def update_physics(self, force, torque, dt):
        """Update the physical state of the robot"""
        # Apply force to change velocity
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        
        # Apply torque to change angular velocity
        angular_acceleration = torque / (self.mass * 0.1)  # Simplified moment of inertia
        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity = np.clip(self.angular_velocity, -2.0, 2.0)  # Limit angular velocity
        
        # Update position and orientation
        self.position += self.velocity * dt
        self.orientation += self.angular_velocity * dt
        
        # Normalize orientation
        self.orientation = ((self.orientation + np.pi) % (2 * np.pi)) - np.pi
    
    def run_control_cycle(self, sensor_data, target_state, dt):
        """Run one complete control cycle"""
        # Perception: Process sensor data
        processed_perception = self.perception_module.process(sensor_data)
        
        # Motion planning: Determine next action
        control_commands = self.motion_planner.plan(processed_perception, self.position, target_state, self.velocity)
        
        # Control: Send commands to actuators
        force_cmd, torque_cmd = self.control_module.compute_control(control_commands, self.velocity)
        
        # Apply actuator limitations
        force_cmd = np.clip(force_cmd, -self.max_force, self.max_force)
        torque_cmd = np.clip(torque_cmd, -self.max_torque, self.max_torque)
        
        # Simulate actuator delay
        time.sleep(self.actuator_delay)  # In sim, just add delay
        
        # Update physics based on commands
        self.update_physics(force_cmd, torque_cmd, dt)
        
        # Safety check
        safety_info = self.safety_system.check_safety(self.position, self.velocity, sensor_data)
        
        # Store data
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.command_history.append((force_cmd, torque_cmd))
        self.error_history.append(safety_info['risk_level'])
        
        return {
            'perception': processed_perception,
            'commands': control_commands,
            'safety': safety_info,
            'state': {'position': self.position.copy(), 'velocity': self.velocity.copy()}
        }
    
    def run_simulation(self, duration=10.0):
        """Run system simulation for specified duration"""
        dt = self.control_loop_period
        num_steps = int(duration / dt)
        
        # Define a simple target trajectory
        target_positions = []
        for t in np.linspace(0, duration, num_steps):
            # Spiral trajectory
            x = 3 * np.cos(0.2 * t)
            y = 3 * np.sin(0.2 * t)
            target_positions.append(np.array([x, y]))
        
        environment_state = {}  # Placeholder for environment
        
        for i in range(num_steps):
            # Get current target
            target = target_positions[i] if i < len(target_positions) else target_positions[-1]
            
            # Simulate environment sensing
            sensor_data = self.sense_environment(environment_state)
            
            # Run control cycle
            result = self.run_control_cycle(sensor_data, target, dt)
            
            # Print progress occasionally
            if i % 50 == 0:
                print(f"Time: {i*dt:.2f}s, Position: ({self.position[0]:.2f}, {self.position[1]:.2f})")
                
        return result

class PerceptionModule:
    """Handles sensor data processing"""
    def __init__(self):
        self.last_obstacles = []
        
    def process(self, raw_sensor_data):
        """Process raw sensor data into meaningful information"""
        # Identify obstacles from sensor data
        min_distance = 0.5  # Minimum distance to consider as obstacle
        
        obstacles = []
        sensor_angles = np.linspace(0, 2*np.pi, len(raw_sensor_data), endpoint=False)
        
        for dist, angle in zip(raw_sensor_data, sensor_angles):
            if dist < min_distance:
                # Convert to Cartesian coordinates relative to robot
                x_obs = dist * np.cos(angle)
                y_obs = dist * np.sin(angle)
                obstacles.append({'distance': dist, 'angle': angle, 'pos': (x_obs, y_obs)})
        
        self.last_obstacles = obstacles
        return {
            'obstacles': obstacles,
            'clear_directions': len(obstacles) == 0,
            'closest_obstacle': min([obs['distance'] for obs in obstacles]) if obstacles else float('inf')
        }

class MotionPlanner:
    """Plans motion based on perception and goals"""
    def __init__(self):
        self.path_history = []
        
    def plan(self, perception, current_pos, target_pos, current_vel):
        """Plan next action based on perception and goal"""
        # Simple obstacle-aware navigation
        obstacle_info = perception['obstacles']
        closest_dist = perception['closest_obstacle']
        
        # Calculate desired direction to target
        desired_vector = target_pos - current_pos
        desired_distance = np.linalg.norm(desired_vector)
        desired_direction = desired_vector / desired_distance if desired_distance > 0 else np.array([1, 0])
        
        # If obstacle is too close, modify direction
        avoidance_vector = np.array([0.0, 0.0])
        if obstacle_info and closest_dist < 1.0:
            # Simple vector field approach: move away from closest obstacle
            closest_obstacle = min(obstacle_info, key=lambda x: x['distance'])
            obstacle_angle = closest_obstacle['angle']
            avoidance_dir = np.array([np.cos(obstacle_angle + np.pi), np.sin(obstacle_angle + np.pi)])
            avoidance_magnitude = 1.0 / max(closest_dist, 0.1)  # Stronger as we get closer
            avoidance_vector = avoidance_magnitude * avoidance_dir
        
        # Combine desired direction with obstacle avoidance
        combined_direction = desired_direction * 0.7 + avoidance_vector * 0.3
        combined_direction = combined_direction / np.linalg.norm(combined_direction)
        
        return {
            'desired_direction': combined_direction,
            'desired_speed': min(0.5, desired_distance * 0.2),  # Slower when close to target
            'avoidance_vector': avoidance_vector
        }

class ControlModule:
    """Converts high-level commands to actuator commands"""
    def __init__(self):
        self.prev_heading_error = 0.0
        self.integral_heading_error = 0.0
        
    def compute_control(self, commands, current_velocity):
        """Convert navigation commands to force and torque commands"""
        desired_direction = commands['desired_direction']
        desired_speed = commands['desired_speed']
        
        # Calculate desired heading
        desired_heading = np.arctan2(desired_direction[1], desired_direction[0])
        
        # Current heading (from velocity)
        current_speed = np.linalg.norm(current_velocity)
        if current_speed > 0.01:
            current_heading = np.arctan2(current_velocity[1], current_velocity[0])
        else:
            # Use robot's orientation when stationary
            current_heading = 0  # Placeholder - in real system, use robot's heading
            
        # Calculate heading error
        heading_error = desired_heading - current_heading
        # Normalize to [-π, π]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
            
        # PID controller for heading
        kp_heading = 2.0
        ki_heading = 0.1
        kd_heading = 0.05
        
        self.integral_heading_error += heading_error
        derivative_heading = (heading_error - self.prev_heading_error) / 0.05  # dt
        
        torque = kp_heading * heading_error + ki_heading * self.integral_heading_error + kd_heading * derivative_heading
        torque = np.clip(torque, -5.0, 5.0)  # Limit torque
        
        # Proportional controller for velocity
        velocity_error = desired_speed - current_speed
        force_magnitude = 10.0 * velocity_error  # Simple proportional control
        force_magnitude = np.clip(force_magnitude, -10.0, 10.0)  # Limit force
        
        # Force direction is in the desired movement direction
        force_vector = force_magnitude * desired_direction
        
        self.prev_heading_error = heading_error
        
        return force_vector, torque

class SafetySystem:
    """Monitors system state for safety issues"""
    def __init__(self):
        self.emergency_stop = False
        self.last_safe_check = time.time()
        
    def check_safety(self, position, velocity, sensor_data):
        """Check current state for safety issues"""
        # Check for critical obstacles too close
        critical_distance = 0.3  # meters
        min_sensor_reading = min(sensor_data) if len(sensor_data) > 0 else float('inf')
        
        risk_level = 0  # 0-1 scale, 1 is highest risk
        if min_sensor_reading < critical_distance:
            risk_level = (critical_distance - min_sensor_reading) / critical_distance
        elif min_sensor_reading < 0.8:
            risk_level = (0.8 - min_sensor_reading) / 0.8 * 0.5  # Lower risk at greater distance
        
        # Check velocity limits
        speed = np.linalg.norm(velocity)
        if speed > 1.0:
            risk_level = max(risk_level, 0.3)  # Increase risk at high speeds
        
        # Determine if emergency stop is needed
        emergency_stop_needed = risk_level > 0.8 or (min_sensor_reading < 0.15 and speed > 0.3)
        
        return {
            'risk_level': risk_level,
            'emergency_stop_needed': emergency_stop_needed,
            'obstacle_detected': min_sensor_reading < 1.0,
            'safe': risk_level < 0.5
        }

# Run the complete system simulation
system = PhysicalAISystem()
print("Starting Physical AI System Simulation...")
result = system.run_simulation(duration=15.0)
print("Simulation completed!")

# Analyze and visualize results
plt.figure(figsize=(20, 15))

# Plot robot trajectory
plt.subplot(3, 3, 1)
trajectory = np.array(system.position_history)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Robot Path')
plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=200, label='Start', zorder=5)
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=200, label='End', zorder=5)
plt.title('Robot Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot velocity over time
plt.subplot(3, 3, 2)
velocities = np.array(system.velocity_history)
speeds = np.linalg.norm(velocities, axis=1)
plt.plot(np.arange(len(speeds)) * system.control_loop_period, speeds, 'g-', linewidth=2)
plt.title('Robot Speed Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True, alpha=0.3)

# Plot x and y position over time
plt.subplot(3, 3, 3)
time_axis = np.arange(len(trajectory)) * system.control_loop_period
plt.plot(time_axis, trajectory[:, 0], label='X Position', linewidth=2)
plt.plot(time_axis, trajectory[:, 1], label='Y Position', linewidth=2)
plt.title('Position Components Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot commanded forces over time
plt.subplot(3, 3, 4)
if system.command_history:
    forces = np.array([cmd[0] for cmd in system.command_history])
    plt.plot(time_axis[:-1], forces[:, 0], label='X Force', linewidth=2)
    plt.plot(time_axis[:-1], forces[:, 1], label='Y Force', linewidth=2)
    plt.title('Commanded Forces Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot commanded torques over time
plt.subplot(3, 3, 5)
if system.command_history:
    torques = [cmd[1] for cmd in system.command_history]
    plt.plot(time_axis[:-1], torques, 'm-', linewidth=2)
    plt.title('Commanded Torque Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N⋅m)')
    plt.grid(True, alpha=0.3)

# Plot safety risk over time
plt.subplot(3, 3, 6)
plt.plot(time_axis[:-1], system.error_history, 'r-', linewidth=2)
plt.axhline(y=0.5, color='orange', linestyle='--', label='Risk Threshold', alpha=0.7)
plt.title('Safety Risk Level Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Risk Level (0-1)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot control loop timing analysis
plt.subplot(3, 3, 7)
if len(system.command_history) > 1:
    plt.hist([np.linalg.norm(cmd[0]) for cmd in system.command_history], bins=30, alpha=0.7, label='Force Magnitudes')
    plt.title('Distribution of Control Force Magnitudes')
    plt.xlabel('Force (N)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

# Plot velocity phase space
plt.subplot(3, 3, 8)
plt.plot(velocities[:, 0], velocities[:, 1], 'purple', linewidth=2)
plt.title('Velocity Phase Space')
plt.xlabel('X Velocity (m/s)')
plt.ylabel('Y Velocity (m/s)')
plt.grid(True, alpha=0.3)

# Plot distance to origin
plt.subplot(3, 3, 9)
distances_to_origin = np.linalg.norm(trajectory, axis=1)
plt.plot(time_axis, distances_to_origin, 'c-', linewidth=2)
plt.title('Distance to Origin Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate performance metrics
final_distance = np.linalg.norm(trajectory[-1])
avg_speed = np.mean(speeds)
max_speed = np.max(speeds)
safety_incidents = sum(1 for risk in system.error_history if risk > 0.5)

print(f"\nSystem Integration Simulation Results:")
print(f"Final position: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})")
print(f"Final distance to origin: {final_distance:.3f} m")
print(f"Average speed: {avg_speed:.3f} m/s")
print(f"Maximum speed: {max_speed:.3f} m/s")
print(f"Safety incidents (>0.5 risk): {safety_incidents}/{len(system.error_history)}")
print(f"Total trajectory length: {np.sum(np.sqrt(np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2)):.3f} m")
```

### Failure Mode Analysis

Now let's implement a failure mode analysis for the system:

```python
import numpy as np
import matplotlib.pyplot as plt
import copy

class FailureModeAnalyzer:
    """Analyzes potential failure modes in Physical AI systems"""
    def __init__(self):
        self.failure_modes = []
        
    def analyze_sensor_failures(self, sensor_config):
        """Analyze what happens when sensors fail"""
        failures = []
        
        for i in range(len(sensor_config)):
            # Simulate single sensor failure
            degraded_config = sensor_config.copy()
            degraded_config[i] = float('inf')  # Sensor fails to infinite distance
            
            failure = {
                'type': 'Sensor Failure',
                'component': f'Sensor_{i}',
                'impact': 'Reduced environment awareness',
                'severity': 'Medium',
                'likelihood': 0.05,  # 5% chance per hour of operation
                'symptoms': f'Reduced perception in sector {i}',
                'mitigation': 'Sensor fusion with redundant sensors, detection algorithms'
            }
            failures.append(failure)
        
        return failures
    
    def analyze_actuator_failures(self, actuator_limits):
        """Analyze what happens when actuators fail"""
        failures = []
        
        # Actuator stuck at position
        failures.append({
            'type': 'Actuator Failure',
            'component': 'Drive Motor',
            'impact': 'Reduced maneuverability',
            'severity': 'High',
            'likelihood': 0.02,  # 2% chance per hour of operation
            'symptoms': 'Robot moves in straight line despite commands',
            'mitigation': 'Actuator health monitoring, emergency stop procedures'
        })
        
        # Actuator provides incorrect force/torque
        failures.append({
            'type': 'Actuator Degradation',
            'component': 'Motor Control',
            'impact': 'Imprecise control',
            'severity': 'Medium',
            'likelihood': 0.1,  # 10% chance per hour due to wear
            'symptoms': 'Drift from intended path, inconsistent speeds',
            'mitigation': 'Regular calibration, feedback control with position sensors'
        })
        
        # Power supply failure
        failures.append({
            'type': 'Power Failure',
            'component': 'Battery/Power Supply',
            'impact': 'Complete system shutdown',
            'severity': 'Critical',
            'likelihood': 0.01,  # 1% chance per hour
            'symptoms': 'Sudden power loss, inability to move/operate sensors',
            'mitigation': 'Backup power systems, safe shutdown procedures'
        })
        
        return failures
    
    def analyze_computational_failures(self):
        """Analyze computational system failures"""
        failures = []
        
        # Processing lag
        failures.append({
            'type': 'Computational Failure',
            'component': 'Control Computer',
            'impact': 'Delayed responses, poor performance',
            'severity': 'High',
            'likelihood': 0.15,  # 15% chance due to computational load
            'symptoms': 'Sluggish response, missed deadlines',
            'mitigation': 'Real-time scheduling, computational load monitoring, simplified fallback algorithms'
        })
        
        # Memory overflow
        failures.append({
            'type': 'Computational Failure',
            'component': 'Memory System',
            'impact': 'System crash',
            'severity': 'Critical',
            'likelihood': 0.03,  # 3% chance per hour of operation
            'symptoms': 'Unresponsive system, restart required',
            'mitigation': 'Memory leak detection, garbage collection, bounds checking'
        })
        
        # Algorithm failure (e.g., planner gets stuck)
        failures.append({
            'type': 'Algorithmic Failure',
            'component': 'Motion Planner',
            'impact': 'Robot stops moving or oscilates',
            'severity': 'Medium',
            'likelihood': 0.08,  # 8% chance in challenging environments
            'symptoms': 'Robot freezes, cycles in place',
            'mitigation': 'Timeout mechanisms, fallback controllers, random walk recovery'
        })
        
        return failures
    
    def analyze_integration_failures(self):
        """Analyze failures that occur at system boundaries"""
        failures = []
        
        # Timing mismatch between components
        failures.append({
            'type': 'Integration Failure',
            'component': 'System Timing',
            'impact': 'Erratic behavior',
            'severity': 'High',
            'likelihood': 0.12,  # 12% chance due to complex timing
            'symptoms': 'Inconsistent responses, control instability',
            'mitigation': 'Synchronized clock systems, deterministic scheduling, buffering'
        })
        
        # Communication failure between components
        failures.append({
            'type': 'Integration Failure',
            'component': 'Communication Bus',
            'impact': 'Partial system failure',
            'severity': 'High',
            'likelihood': 0.05,  # 5% chance per hour
            'symptoms': 'Components stop responding, sensor/actuator disconnection',
            'mitigation': 'Redundant communication paths, heartbeat protocols, graceful degradation'
        })
        
        return failures
    
    def generate_fmea(self, operation_hours=100):
        """Generate Failure Modes and Effects Analysis (FME&A)"""
        print(f"Generating FME&A for {operation_hours} hours of operation...")
        
        # Gather all failure modes
        sensor_failures = self.analyze_sensor_failures([1.0] * 16)  # 16 sensors
        actuator_failures = self.analyze_actuator_failures({'max_force': 10.0, 'max_torque': 5.0})
        computational_failures = self.analyze_computational_failures()
        integration_failures = self.analyze_integration_failures()
        
        all_failures = (sensor_failures + actuator_failures + 
                       computational_failures + integration_failures)
        
        # Calculate expected occurrences over operation period
        for failure in all_failures:
            failure['expected_occurrences'] = failure['likelihood'] * operation_hours
            # Risk Priority Number (RPN) = Severity * Occurrence * Detection
            severity_map = {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 7}
            failure['rpn'] = severity_map[failure['severity']] * failure['likelihood'] * 5  # Detection = 5 (hard to detect)
        
        # Sort by RPN
        all_failures.sort(key=lambda x: x['rpn'], reverse=True)
        
        return all_failures

# Create analyzer and run analysis
analyzer = FailureModeAnalyzer()
failure_modes = analyzer.generate_fmea(operation_hours=100)

# Display the results
print("\nFAILURE MODES ANALYSIS RESULTS")
print("="*60)

high_risk_failures = [fm for fm in failure_modes if 'Critical' in fm['severity'] or 'High' in fm['severity']]

print(f"Total identified failure modes: {len(failure_modes)}")
print(f"High/Critical risk failures: {len(high_risk_failures)}")
print("\nTop 10 High-Risk Failures:")
print("-" * 60)

for i, fm in enumerate(failure_modes[:10]):
    print(f"{i+1}. {fm['type']} - {fm['component']}")
    print(f"   Impact: {fm['impact']}")
    print(f"   Severity: {fm['severity']}")
    print(f"   Expected occurrences ({fm['expected_occurrences']:.2f} per 100h)")
    print(f"   Risk Priority: {fm['rpn']:.2f}")
    print(f"   Mitigation: {fm['mitigation']}")
    print()

# Visualize failure modes
plt.figure(figsize=(15, 10))

# Group failures by category
categories = ['Sensor', 'Actuator', 'Computational', 'Integration']
counts = {
    'Sensor': len([fm for fm in failure_modes if 'Sensor' in fm['type']]),
    'Actuator': len([fm for fm in failure_modes if 'Actuator' in fm['type']]),
    'Computational': len([fm for fm in failure_modes if 'Computational' in fm['type']]),
    'Integration': len([fm for fm in failure_modes if 'Integration' in fm['type']])
}

# Plot pie chart of failure types
plt.subplot(2, 3, 1)
plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Failure Types')

# Plot severity levels
severity_counts = {
    'Critical': len([fm for fm in failure_modes if fm['severity'] == 'Critical']),
    'High': len([fm for fm in failure_modes if fm['severity'] == 'High']),
    'Medium': len([fm for fm in failure_modes if fm['severity'] == 'Medium']),
    'Low': len([fm for fm in failure_modes if fm['severity'] == 'Low'])
}

plt.subplot(2, 3, 2)
plt.bar(severity_counts.keys(), severity_counts.values())
plt.title('Failure Severity Distribution')
plt.xlabel('Severity Level')
plt.ylabel('Count')

# Plot expected occurrences
plt.subplot(2, 3, 3)
types = [fm['type'] + ' - ' + fm['component'][:10] for fm in failure_modes[:8]]  # Top 8
occurrences = [fm['expected_occurrences'] for fm in failure_modes[:8]]
plt.barh(types, occurrences)
plt.title('Expected Failure Occurrences (per 100 hours)')
plt.xlabel('Expected Occurrences')

# Plot RPN scores
plt.subplot(2, 3, 4)
rpn_values = [fm['rpn'] for fm in failure_modes[:15]]  # Top 15
plt.plot(range(len(rpn_values)), rpn_values, 'ro-')
plt.title('Risk Priority Numbers (Top 15)')
plt.xlabel('Failure Mode Rank')
plt.ylabel('RPN')

# Severity vs Likelihood scatter
plt.subplot(2, 3, 5)
severity_nums = [1 if fm['severity'] == 'Low' else 
                 3 if fm['severity'] == 'Medium' else 
                 5 if fm['severity'] == 'High' else 7 for fm in failure_modes]
plt.scatter([fm['likelihood'] for fm in failure_modes], severity_nums, 
           c=[fm['rpn'] for fm in failure_modes], s=100, alpha=0.7, cmap='viridis')
plt.xlabel('Likelihood')
plt.ylabel('Severity Level (1-7)')
plt.title('Severity vs Likelihood (color = RPN)')

# Likelihood distribution
plt.subplot(2, 3, 6)
likelihoods = [fm['likelihood'] for fm in failure_modes]
plt.hist(likelihoods, bins=10, edgecolor='black')
plt.xlabel('Likelihood (failures per hour)')
plt.ylabel('Frequency')
plt.title('Distribution of Failure Likelihoods')

plt.tight_layout()
plt.show()

print("\nRecommended Design Improvements Based on Analysis:")
print("-"*60)
print("1. Implement redundant sensors with fusion algorithms")
print("2. Add real-time monitoring of computational load")
print("3. Design graceful degradation modes for high-risk scenarios")
print("4. Implement heartbeat and watchdog systems for communication integrity")
print("5. Add fallback controllers for critical actuator failures")
```

## Real-World Mapping

### Software-Hardware Integration in Production Systems

Real-world Physical AI systems demonstrate these integration principles:

| Component | Integration Requirement | Typical Solution |
|-----------|------------------------|------------------|
| **Sensors** | Low-latency, high-accuracy data processing | Dedicated hardware accelerators, edge computing |
| **Actuators** | Precise control with hardware limitations | Model-predictive control, safety-rated drives |
| **Computing** | Real-time performance with power constraints | Specialized AI chips, optimized algorithms |
| **Communication** | Reliable, fast component coordination | TSN networks, shared memory systems |
| **Safety** | Fail-safe operation at all times | Redundant systems, safety-rated controllers |

### Common Failure Modes in Practice

| Failure Category | Common Causes | Prevention Strategies |
|------------------|---------------|----------------------|
| **Sensor** | Dust, moisture, vibration, electromagnetic interference | Protected housings, cleaning systems, redundancy |
| **Actuator** | Wear, overheating, power fluctuations | Condition monitoring, derating, backup systems |
| **Processing** | Memory leaks, algorithm divergence, thermal issues | Watchdogs, heap monitoring, thermal management |
| **Communication** | Cable damage, protocol errors, bandwidth limits | Shielded cables, error detection, redundancy |
| **Integration** | Timing mismatches, data corruption, resource contention | Real-time OS, buffered communication, schedulers |

### Industry Best Practices

1. **Design for Failure**: Assume components will fail and design graceful degradation
2. **Modular Architecture**: Isolate failures to prevent cascading effects
3. **Extensive Testing**: Test not just individual components but integration points
4. **Monitoring**: Continuous monitoring of system health and performance
5. **Documentation**: Clear documentation of integration points and dependencies

### Debugging Techniques

| Technique | Application | Benefits |
|-----------|-------------|----------|
| **State Logging** | All system components | Comprehensive post-mortem analysis |
| **Hardware-in-the-Loop** | Control algorithm validation | Validates software-hardware integration |
| **Unit Testing** | Individual components | Early defect detection |
| **Integration Testing** | Subsystem combinations | Identifies interface issues |
| **System Testing** | Complete system | Validates overall functionality |

## Exercises

### Beginner Tasks
1. Run the system integration simulation and observe how different components interact
2. Modify sensor noise levels and observe the effect on robot behavior
3. Change the control loop timing and see how it affects performance
4. Run the failure mode analysis to understand potential system vulnerabilities

### Stretch Challenges
1. Implement a watchdog system in the simulation to detect failures
2. Add a second robot and implement coordination between them
3. Design a recovery routine for when the robot gets trapped

## Summary

This chapter explored how to think about software and hardware as integrated systems in Physical AI. We implemented a complete system simulation that demonstrates the connections between perception, control, and actuation, and analyzed potential failure modes that can occur at the integration points.

The key to successful Physical AI systems is understanding that software and hardware are not separate components but parts of an interconnected system where the behavior of one affects the others. This holistic view is essential for designing robust systems that can operate safely in the physical world.

In the next chapter, we'll explore the ethical considerations and future directions for Physical AI.