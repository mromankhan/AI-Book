---
sidebar_position: 1
---

# Chapter 1.1: Anatomy of a Robot

## Overview

In this chapter, you will learn:
- The fundamental components that make up a robot
- The concept of degrees of freedom and how they affect robot motion
- The differences between humanoid and non-humanoid robots
- How sensors, actuators, and controllers work together

Understanding the basic anatomy of robots is essential to Physical AI, as it forms the foundation for how AI systems interact with physical systems.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Identify the three fundamental components of any robot (sensors, actuators, controllers)
2. Explain the concept of degrees of freedom and its impact on robot capabilities
3. Distinguish between humanoid and non-humanoid robot designs
4. Understand how sensors, actuators, and controllers work together in a complete system

### Why This Matters

Understanding robot anatomy is fundamental to Physical AI because the AI system must be designed to work with the specific physical constraints and capabilities of the robotic platform. The sensors determine what information is available to the AI, the actuators determine how the AI can affect the physical world, and the controllers determine how these components interact. This knowledge is essential for creating effective Physical AI systems.

## Core Concepts

### Sensors, Actuators, Controllers

The three fundamental components of any robot:

1. **Sensors**: Provide the robot with information about its environment and internal state.
   - Internal sensors: encoders (position/velocity), IMUs (orientation/acceleration), current/torque sensors
   - External sensors: cameras, LiDAR, ultrasonic, touch sensors, etc.

2. **Actuators**: Enable the robot to interact with the physical world.
   - Motors: DC motors, stepper motors, servo motors, etc.
   - Linear actuators: pneumatic, hydraulic, or electric linear actuators
   - Specialized actuators: shape memory alloys, pneumatic muscles

3. **Controllers**: Process sensor data and send commands to actuators.
   - Low-level controllers: motor controllers, PID controllers
   - High-level controllers: behavior-based controllers, AI planning systems
   - Distributed vs centralized control architectures

### Degrees of Freedom (DOF)

Degrees of Freedom refers to the number of independent movements a mechanical system can perform. For robots, it indicates:

- The number of joints in a robotic arm or leg
- Each rotational joint contributes 1 DOF (rotation about an axis)
- Each prismatic joint contributes 1 DOF (linear motion along an axis)
- A 6-DOF system can position its end-effector anywhere in 3D space with any orientation
- Redundant DOF (more than required) provides flexibility and obstacle avoidance

### Humanoid vs Non-Humanoid Robots

**Humanoid Robots**:
- Designed with human-like form factor (bipedal, arms, head)
- Advantages: Intuitive human-robot interaction, compatibility with human environments
- Challenges: Complex balance, high number of DOFs, energy consumption
- Examples: Honda ASIMO, Boston Dynamics Atlas, SoftBank Pepper

**Non-Humanoid Robots**:
- Designed for specific tasks or environments
- Advantages: Specialized for efficiency, simpler control systems
- Examples: Industrial arms, wheeled robots, drones, snake robots, robotic vacuum cleaners

## Hands-on Section

### Building a Simple Robotic Arm Simulation

Let's create a simulation of a basic 2-DOF robotic arm using PyBullet:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Add gravity
p.setGravity(0, 0, -9.8)

# Load plane
p.loadURDF("plane.urdf")

# Create a simple robotic arm (2 DOF)
# Define base position and orientation
baseStartPos = [0, 0, 0]
baseStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

# Load the URDF (we'll create a simple arm from basic shapes)
# Create a simple arm with 2 segments
# Base
baseId = p.loadURDF("cube.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True, globalScaling=0.5)

# Upper arm
upperArmId = p.loadURDF("cube.urdf", [0.25, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=False, globalScaling=0.25)

# Lower arm
lowerArmId = p.loadURDF("cube.urdf", [0.5, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]),
                        useFixedBase=False, globalScaling=0.25)

# Create constraints to connect the segments like a robotic arm
# Connect base to upper arm
base_to_upper = p.createConstraint(baseId, -1, upperArmId, -1,
                                   p.JOINT_REVOLUTE, [0, 0, 1], [0, 0, 0], [0.25, 0, 0])

# Connect upper arm to lower arm
upper_to_lower = p.createConstraint(upperArmId, -1, lowerArmId, -1,
                                    p.JOINT_REVOLUTE, [0, 0, 1], [0.5, 0, 0], [0.25, 0, 0])

# Control the joints to move the arm
# Get joint information
num_joints = p.getNumJoints(upperArmId)
print(f"Number of joints in upper arm: {num_joints}")

# Add controls for the joints
joint1_id = 0  # First joint connects upper arm to base
joint2_id = 1  # Second joint connects lower arm to upper arm

# Enable torque control for joints
p.setJointMotorControl2(upperArmId, joint1_id, p.POSITION_CONTROL, targetPosition=0.5)
p.setJointMotorControl2(lowerArmId, joint2_id, p.POSITION_CONTROL, targetPosition=0.3)

# Run simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
    
    # Change joint positions periodically to make the arm move
    if i % 1000 == 0:
        # Alternate between two positions
        pos1 = 0.5 if i % 2000 == 0 else -0.5
        pos2 = 0.3 if i % 2000 == 0 else -0.3
        p.setJointMotorControl2(upperArmId, joint1_id, p.POSITION_CONTROL, targetPosition=pos1)
        p.setJointMotorControl2(lowerArmId, joint2_id, p.POSITION_CONTROL, targetPosition=pos2)

# Disconnect
p.disconnect()
```

### Advanced Robotic Arm Control

Let's enhance our robotic arm example to include more realistic control and sensor feedback:

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to physics server
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Add gravity
p.setGravity(0, 0, -9.8)

# Load plane
p.loadURDF("plane.urdf")

# Create a target object
target_pos = [0.7, 0.5, 0.1]
target_orn = p.getQuaternionFromEuler([0, 0, 0])
target_id = p.loadURDF("cube.urdf", target_pos, target_orn,
                      useFixedBase=True, globalScaling=0.1)

# Create a more complex arm
# Base (fixed)
base_id = p.loadURDF("cube.urdf", [0, 0, 0.1],
                     p.getQuaternionFromEuler([0, 0, 0]),
                     useFixedBase=True, globalScaling=0.2)

# Upper arm
upper_arm_start_pos = [0.15, 0, 0.2]
upper_arm_id = p.loadURDF("cube.urdf", upper_arm_start_pos,
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False, globalScaling=0.1)

# Lower arm
lower_arm_start_pos = [0.3, 0, 0.2]
lower_arm_id = p.loadURDF("cube.urdf", lower_arm_start_pos,
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False, globalScaling=0.1)

# Hand/end effector
hand_start_pos = [0.45, 0, 0.2]
hand_id = p.loadURDF("cube.urdf", hand_start_pos,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=False, globalScaling=0.08)

# Create constraints to connect the segments
# Base to upper arm
base_to_upper = p.createConstraint(base_id, -1, upper_arm_id, -1,
                                   p.JOINT_REVOLUTE, [0, 0, 1],
                                   [0.15, 0, 0.2], [0.15, 0, 0.2])

# Upper arm to lower arm
upper_to_lower = p.createConstraint(upper_arm_id, -1, lower_arm_id, -1,
                                    p.JOINT_REVOLUTE, [0, 0, 1],
                                    [0.3, 0, 0.2], [0.15, 0, 0.2])

# Lower arm to hand
lower_to_hand = p.createConstraint(lower_arm_id, -1, hand_id, -1,
                                   p.JOINT_REVOLUTE, [0, 0, 1],
                                   [0.45, 0, 0.2], [0.15, 0, 0.2])

# Sensor simulation: Read joint positions and end-effector position
def get_joint_positions():
    # Get the joint states for the upper arm (child of base)
    joint_states = p.getJointState(upper_arm_id, 0)
    upper_joint_pos = joint_states[0]

    # Get the joint states for the lower arm (child of upper arm)
    joint_states = p.getJointState(lower_arm_id, 0)
    lower_joint_pos = joint_states[0]

    return upper_joint_pos, lower_joint_pos

def get_end_effector_position():
    pos, orn = p.getBasePositionAndOrientation(hand_id)
    return pos

def move_to_target(current_pos, target_pos, step_size=0.01):
    direction = np.array(target_pos) - np.array(current_pos)
    distance = np.linalg.norm(direction)

    if distance < step_size:
        return target_pos

    direction = direction / distance  # Normalize
    new_pos = current_pos + direction * step_size
    return new_pos

# Control parameters
prev_error = [0, 0]  # For PID controller
integral = [0, 0]
kp, ki, kd = 1.5, 0.01, 0.1  # PID gains

# Run simulation with control
for i in range(20000):
    p.stepSimulation()

    # Get current end effector position
    effector_pos = get_end_effector_position()

    # Get joint angles
    joint1_pos, joint2_pos = get_joint_positions()

    # Calculate error to target
    error = np.array(target_pos) - np.array(effector_pos)

    # Simple PID controller for joint control
    # Calculate PID terms for each joint
    derivative = [(error[j] - prev_error[j]) for j in range(2)]
    integral = [integral[j] + error[j] for j in range(2)]

    # Calculate control signals (simplified)
    control_signal = [kp * error[j] + ki * integral[j] + kd * derivative[j]
                      for j in range(2)]

    # Apply control with limits
    control_signal[0] = max(-1, min(1, control_signal[0]))
    control_signal[1] = max(-1, min(1, control_signal[1]))

    # Convert to joint angle changes (simplified)
    target_joint1 = joint1_pos + control_signal[0] * 0.01
    target_joint2 = joint2_pos + control_signal[1] * 0.01

    # Apply joint control
    p.setJointMotorControl2(upper_arm_id, 0, p.POSITION_CONTROL,
                           targetPosition=target_joint1)
    p.setJointMotorControl2(lower_arm_id, 0, p.POSITION_CONTROL,
                           targetPosition=target_joint2)

    # Store current error for next iteration
    prev_error = error

    if i % 1000 == 0:
        print(f"Step {i}: EE Pos: ({effector_pos[0]:.2f}, {effector_pos[1]:.2f}) "
              f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}) "
              f"Distance: {np.sqrt(error[0]**2 + error[1]**2):.2f}")

    time.sleep(1./240.)

p.disconnect()
```

This advanced example demonstrates:
- More complex robot structure with an end effector
- Joint position sensors to provide feedback
- PID control algorithm to move the robot toward a target
- Position tracking and error calculation

### Observing Physical Constraints

1. Run the simulation and observe how the arm moves.
2. Notice how the joints limit the movement to specific planes (since we're using revolute joints).
3. Try changing the joint limits and observe how it affects the robot's workspace.
4. Add a "payload" (small cube) at the end effector position and observe how it affects the system dynamics.

## Real-World Mapping

Real robotic systems face additional challenges not fully captured in simulation:

- **Joint Friction**: Real joints have friction that affects motion
- **Actuator Dynamics**: Motors have limitations on speed, torque, and acceleration
- **Flexibility**: Links may flex under load, requiring more complex control strategies
- **Calibration**: Real sensors need calibration to provide accurate measurements
- **Safety Systems**: Real robots need safety systems to prevent damage

Industrial robotics uses sophisticated control systems that account for these real-world factors, often employing:
- Advanced control algorithms (adaptive control, model predictive control)
- Sophisticated sensors for precise positioning
- Safety-rated controllers designed for human-robot collaboration

### Simulation vs. Real Robot Comparison

| Component | Simulation | Real Robot |
|-----------|------------|------------|
| **Sensors** | Perfect, noise-free measurements | Noisy, with bias, drift, and limited range |
| **Actuators** | Instantaneous torque/position control | Limited torque, speed constraints, thermal limits |
| **Joints** | Ideal revolute/prismatic joints | Friction, backlash, wear over time |
| **Environment** | Deterministic physics | Unpredictable disturbances, changing conditions |
| **Safety** | No consequence for errors | Potential damage to robot or environment |
| **Cost** | Minimal computational cost | High cost per test, wear and tear |

### Simulation Strategies for Real-World Transfer

#### Adding Realism to Simulation
1. **Noise Models**: Adding sensor and actuator noise to match real systems
2. **Latency Simulation**: Adding communication delays that exist in real systems
3. **Parameter Variation**: Testing with slightly different physical parameters
4. **Disturbance Forces**: Adding external forces to simulate real-world unpredictability

#### Control Techniques for Simulation-to-Reality Transfer
1. **Robust Control**: Controllers designed to work despite model uncertainty
2. **Adaptive Control**: Controllers that can adjust parameters based on observations
3. **Learning-based Control**: Using machine learning to adapt controllers
4. **Gain Scheduling**: Adjusting controller parameters based on operating conditions

## Exercises

### Beginner Tasks
1. Run the provided robotic arm simulation code.
2. Modify the joint positions to make the arm move to a specific location.
3. Change the colors or sizes of the arm segments to customize the simulation.
4. Add a target object in the simulation and try to make the arm reach toward it.

### Stretch Challenges
1. Implement inverse kinematics to calculate the joint angles needed for the end effector to reach a specific position.
2. Add a 3rd DOF to the robotic arm and implement a more complex control pattern.
3. Create a trajectory planning system that moves the arm smoothly between points rather than jumping between positions.

## Summary

This chapter introduced the fundamental components of robots: sensors, actuators, and controllers. We discussed the important concept of degrees of freedom and the differences between humanoid and non-humanoid robotic designs.

Understanding robot anatomy is crucial for Physical AI, as AI systems must account for the physical constraints and capabilities of the robotic platform they're controlling.

In the next chapters, we'll dive deeper into specific components, starting with sensors and perception.