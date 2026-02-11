---
sidebar_position: 3
---

# Chapter 1.3: Actuators & Motion

## Overview

In this chapter, you will learn:
- The different types of actuators used in robotics
- How torque and speed trade-offs affect robot motion
- The basics of joint control
- Safety considerations when working with actuators

Actuators are the muscles of Physical AI systems, converting electrical signals into physical motion that allows robots to interact with the world.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Identify and describe the main types of actuators used in robotics (motors, linear actuators, etc.)
2. Explain the fundamental torque-speed trade-off in all actuators
3. Implement basic joint control strategies (position, velocity, torque control)
4. Apply safety considerations when designing actuator-based systems

### Why This Matters

Actuators are the interface between AI decision-making and physical action. Understanding actuators and their limitations is essential for creating Physical AI systems that can effectively and safely interact with the physical world. The choice and control of actuators directly impacts a robot's ability to perform tasks and its safety in human environments.

## Core Concepts

### Motors

Motors are the most common actuators in robotics:

- **DC Motors**: Simple and inexpensive, but require additional control circuitry for precise positioning
- **Stepper Motors**: Move in discrete steps, good for precise positioning without feedback
- **Servo Motors**: Include position feedback and control circuitry, ideal for precise control
- **Brushless DC Motors**: More efficient and longer-lasting than brushed motors

### Torque vs Speed

A fundamental trade-off in all actuators is between torque and speed:

- **High Torque**: Required for lifting heavy objects, but typically results in slower motion
- **High Speed**: Allows for fast movement, but with reduced force capacity
- **Gear Ratios**: Used to adjust the balance between torque and speed
- **Motor Constants**: Each motor has inherent characteristics that define this trade-off

### Joint Control Basics

Controlling robot joints involves several key concepts:

- **Position Control**: Moving to a specific angle or position
- **Velocity Control**: Moving at a specific speed
- **Torque Control**: Applying a specific force or torque
- **Impedance Control**: Controlling the stiffness or compliance of the joint

### Common Actuator Types

- **Rotary Actuators**: Provide rotational motion (most common in articulated robots)
- **Linear Actuators**: Provide straight-line motion (pneumatic, hydraulic, or electric)
- **Smart Actuators**: Include integrated control and feedback capabilities
- **Series Elastic Actuators**: Include springs to provide compliant motion

### Safety Considerations

Actuators can pose risks that must be carefully managed:

- **Mechanical Safety**: Preventing harmful collisions or pinching
- **Electrical Safety**: Proper power management and protection
- **Emergency Stops**: Ability to immediately halt motion
- **Force Limiting**: Mechanisms to prevent excessive force application
- **Range Limiting**: Preventing motion beyond safe mechanical limits

## Hands-on Section

### Controlling Motor Motion in Simulation

Let's explore how to control motors and joints in a simulated environment:

```python
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up environment
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create a simple robot arm with multiple joints
# Base (fixed)
base_id = p.loadURDF("cube.urdf", [0, 0, 0.1], 
                     p.getQuaternionFromEuler([0, 0, 0]),
                     useFixedBase=True, globalScaling=0.3)

# Lower arm (larger, heavier segment)
lower_arm_start_pos = [0.2, 0, 0.2]
lower_arm_id = p.loadURDF("cube.urdf", lower_arm_start_pos, 
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False, globalScaling=0.15)

# Upper arm (smaller, lighter segment)
upper_arm_start_pos = [0.4, 0, 0.2]
upper_arm_id = p.loadURDF("cube.urdf", upper_arm_start_pos, 
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False, globalScaling=0.1)

# End effector
end_effector_start_pos = [0.6, 0, 0.2]
end_effector_id = p.loadURDF("cube.urdf", end_effector_start_pos, 
                            p.getQuaternionFromEuler([0, 0, 0]),
                            useFixedBase=False, globalScaling=0.08)

# Create constraints to connect the segments
# Base to lower arm
base_to_lower = p.createConstraint(base_id, -1, lower_arm_id, -1,
                                   p.JOINT_REVOLUTE, [0, 0, 1], 
                                   [0.2, 0, 0.2], [0, 0, 0])

# Lower arm to upper arm
lower_to_upper = p.createConstraint(lower_arm_id, -1, upper_arm_id, -1,
                                    p.JOINT_REVOLUTE, [0, 0, 1], 
                                    [0.4, 0, 0.2], [0.2, 0, 0.2])

# Upper arm to end effector
upper_to_end = p.createConstraint(upper_arm_id, -1, end_effector_id, -1,
                                  p.JOINT_REVOLUTE, [0, 0, 1], 
                                  [0.6, 0, 0.2], [0.2, 0, 0.2])

def move_to_position(joint_index, body_id, target_position, max_force=100):
    """
    Move a joint to a specific position using position control
    """
    p.setJointMotorControl2(bodyIndex=body_id,
                           jointIndex=joint_index,
                           controlMode=p.POSITION_CONTROL,
                           targetPosition=target_position,
                           force=max_force)

def apply_torque(joint_index, body_id, torque):
    """
    Apply a specific torque to a joint
    """
    p.setJointMotorControl2(bodyIndex=body_id,
                           jointIndex=joint_index,
                           controlMode=p.TORQUE_CONTROL,
                           force=torque)

def get_joint_state(joint_index, body_id):
    """
    Get the state of a joint (position, velocity, forces)
    """
    return p.getJointState(body_id, joint_index)

def get_end_effector_position():
    """
    Get the position of the end effector
    """
    pos, orn = p.getBasePositionAndOrientation(end_effector_id)
    return pos

print("Number of joints in lower arm:", p.getNumJoints(lower_arm_id))
print("Number of joints in upper arm:", p.getNumJoints(upper_arm_id))
print("Number of joints in end effector:", p.getNumJoints(end_effector_id))

# Set up some targets
targets = [
    0.5,  # Target for lower arm joint
    -0.3, # Target for upper arm joint
    0.8   # Target for end effector joint
]

# Try to move the arm using position control
for i in range(5000):
    p.stepSimulation()
    
    # Apply position control to the joints
    # Lower arm joint (index 0 in its body)
    if p.getNumJoints(lower_arm_id) > 0:
        move_to_position(0, lower_arm_id, targets[0], max_force=50)
    
    # Upper arm joint (index 0 in its body)
    if p.getNumJoints(upper_arm_id) > 0:
        move_to_position(0, upper_arm_id, targets[1], max_force=30)
    
    # End effector joint (index 0 in its body)
    if p.getNumJoints(end_effector_id) > 0:
        move_to_position(0, end_effector_id, targets[2], max_force=10)
    
    if i % 500 == 0:
        # Print position of the end effector
        ee_pos = get_end_effector_position()
        print(f"Step {i}: End effector position: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")
    
    time.sleep(1./240.)

print("Final end effector position:", get_end_effector_position())
p.disconnect()
```

### Implementing Different Control Strategies

Let's implement different actuator control strategies to understand the torque-speed trade-off:

```python
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time

# Connect to physics server
physicsClient = p.connect(p.DIRECT)  # Use DIRECT for faster computation

# Set up environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create two different pendulums to demonstrate torque-speed trade-offs
# Pendulum 1: High torque, low speed (heavier load)
pendulum1_base = p.loadURDF("cube.urdf", [0, 0, 1], 
                           p.getQuaternionFromEuler([0, 0, 0]),
                           useFixedBase=True, globalScaling=0.2)

pendulum1_body = p.loadURDF("sphere2.urdf", [0, 0.5, 1], 
                           p.getQuaternionFromEuler([0, 0, 0]),
                           useFixedBase=False, globalScaling=0.3)

# Pendulum 2: Lower torque, higher speed (lighter load)
pendulum2_base = p.loadURDF("cube.urdf", [1, 0, 1], 
                           p.getQuaternionFromEuler([0, 0, 0]),
                           useFixedBase=True, globalScaling=0.2)

pendulum2_body = p.loadURDF("sphere2.urdf", [1, 0.3, 1], 
                           p.getQuaternionFromEuler([0, 0, 0]),
                           useFixedBase=False, globalScaling=0.1)

# Create joints for the pendulums
joint1 = p.createConstraint(pendulum1_base, -1, pendulum1_body, -1,
                            p.JOINT_REVOLUTE, [0, 0, 1], [0, 0.5, 1], [0, 0, 0])

joint2 = p.createConstraint(pendulum2_base, -1, pendulum2_body, -1,
                            p.JOINT_REVOLUTE, [0, 0, 1], [1, 0.3, 1], [0, 0, 0])

def apply_control_signal(body_id, joint_index, target_angle, kp=1.0, max_force=10):
    """
    Apply a simple proportional controller to move toward target angle
    """
    joint_state = p.getJointState(body_id, joint_index)
    current_angle = joint_state[0]
    velocity = joint_state[1]
    
    # Calculate error
    error = target_angle - current_angle
    
    # Apply control (proportional control)
    control_signal = kp * error - 0.1 * velocity  # Include damping
    
    # Apply motor control with force limit
    p.setJointMotorControl2(
        bodyIndex=body_id,
        jointIndex=joint_index,
        controlMode=p.TORQUE_CONTROL,
        force=max(-max_force, min(max_force, control_signal))
    )

# Simulate different control scenarios
time_steps = 2000
data_log = {'time': [], 'pos1': [], 'vel1': [], 'pos2': [], 'vel2': [], 'torque1': [], 'torque2': []}

# Start with pendulums at different initial angles
p.resetJointState(pendulum1_body, 0, targetValue=1.0)
p.resetJointState(pendulum2_body, 0, targetValue=1.0)

for i in range(time_steps):
    # Use a sinusoidal target to see how each pendulum responds
    target_angle = 0.5 * np.sin(0.01 * i)
    
    # Apply control to both pendulums with different force limits
    # Pendulum 1: Higher force limit (more torque)
    apply_control_signal(pendulum1_body, 0, target_angle, kp=2.0, max_force=20)
    
    # Pendulum 2: Lower force limit (less torque)
    apply_control_signal(pendulum2_body, 0, target_angle, kp=2.0, max_force=5)
    
    p.stepSimulation()
    
    # Log data
    if i % 10 == 0:  # Log every 10 steps to reduce data volume
        state1 = p.getJointState(pendulum1_body, 0)
        state2 = p.getJointState(pendulum2_body, 0)
        
        data_log['time'].append(i)
        data_log['pos1'].append(state1[0])
        data_log['vel1'].append(state1[1])
        data_log['pos2'].append(state2[0])
        data_log['vel2'].append(state2[1])
        # Torque is not directly readable in this mode, so we'll use a proxy based on control effort
        data_log['torque1'].append(20 if abs(state1[0] - target_angle) > 0.1 else 5)
        data_log['torque2'].append(5 if abs(state2[0] - target_angle) > 0.1 else 1)

# Convert to numpy arrays for plotting
for key in data_log:
    data_log[key] = np.array(data_log[key])

# Plot the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(data_log['time'], data_log['pos1'], label='Pendulum 1 (High Torque)')
plt.plot(data_log['time'], data_log['pos2'], label='Pendulum 2 (Low Torque)')
plt.plot(data_log['time'], 0.5 * np.sin(0.01 * data_log['time']), 'k--', label='Target', alpha=0.5)
plt.title('Position Comparison')
plt.xlabel('Time Step')
plt.ylabel('Position (rad)')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(data_log['time'], data_log['vel1'], label='Pendulum 1 (High Torque)')
plt.plot(data_log['time'], data_log['vel2'], label='Pendulum 2 (Low Torque)')
plt.title('Velocity Comparison')
plt.xlabel('Time Step')
plt.ylabel('Velocity (rad/s)')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(data_log['time'], np.abs(data_log['pos1'] - 0.5 * np.sin(0.01 * data_log['time'])), label='Pendulum 1 Error')
plt.plot(data_log['time'], np.abs(data_log['pos2'] - 0.5 * np.sin(0.01 * data_log['time'])), label='Pendulum 2 Error')
plt.title('Tracking Error')
plt.xlabel('Time Step')
plt.ylabel('Error (rad)')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(data_log['pos1'], data_log['vel1'], label='Pendulum 1 (High Torque)')
plt.title('Phase Space - Pendulum 1')
plt.xlabel('Position (rad)')
plt.ylabel('Velocity (rad/s)')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(data_log['pos2'], data_log['vel2'], label='Pendulum 2 (Low Torque)')
plt.title('Phase Space - Pendulum 2')
plt.xlabel('Position (rad)')
plt.ylabel('Velocity (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()

p.disconnect()

print("Comparison completed. Notice how the higher-torque pendulum (Pendulum 1) is better able to follow the target,")
print("while the lower-torque pendulum (Pendulum 2) shows more tracking error, especially during rapid changes.")
```

## Real-World Mapping

Real actuators have many more limitations and characteristics than simulated ones:

- **Backlash**: Mechanical gaps that cause delays in motion
- **Friction**: Static and dynamic friction affecting precision
- **Heat Generation**: Motors heat up during operation, affecting performance
- **Power Requirements**: Different actuators have different power needs
- **Back-Driveability**: Some actuators allow back-driving, others don't
- **Precision Limits**: Physical limitations on positioning accuracy

Professional robotics systems use sophisticated control techniques to handle these real-world constraints:
- **Feedforward Control**: Predicting needed forces/torques based on motion dynamics
- **Adaptive Control**: Adjusting control parameters based on changing conditions
- **Gain Scheduling**: Using different control parameters for different operating conditions
- **Safety Controllers**: Ensuring safe operation under all conditions

## Exercises

### Beginner Tasks
1. Run the motor control simulation and observe how the arm moves to target positions.
2. Change the max_force parameters in the simulation to see how it affects motion speed and precision.
3. Modify the target angles in the simulation to make the arm trace geometric shapes.
4. Run the torque-speed trade-off simulation and observe the differences between high and low torque systems.

### Stretch Challenges
1. Implement a PID controller instead of the simple proportional controller in the simulation.
2. Design a trajectory that moves the end-effector in a smooth path through multiple waypoints.
3. Create a simulation that demonstrates the effect of backlash by adding a small dead zone in the joint motion.

## Summary

This chapter covered the fundamental actuator types used in robotics: motors, linear actuators, and smart actuators. We explored the crucial torque-speed trade-off that affects all physical systems and examined different control strategies.

Understanding actuators is essential for Physical AI systems, as they determine how robots can interact with and manipulate the physical world. The choice of actuators significantly impacts a robot's capabilities and limitations.

In the next part of this book, we'll explore Physical AI core concepts including perception in more detail, control systems, and decision-making algorithms.