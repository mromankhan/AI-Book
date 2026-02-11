---
sidebar_position: 3
---

# Chapter 2.3: Decision Making

## Overview

In this chapter, you will learn:
- The fundamentals of rule-based systems for robotic decision making
- How finite state machines model robot behaviors
- The concept of behavior trees for complex task planning
- How to implement decision-making algorithms in simulation

Decision making is the cognitive layer of Physical AI, determining what goals to pursue and how to achieve them based on current perceptions and environmental context.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Design rule-based systems for simple robotic tasks
2. Implement finite state machines to model robot behaviors
3. Understand the structure and advantages of behavior trees
4. Simulate decision-making processes in robotic scenarios

### Why This Matters

Decision making bridges perception and action, determining what a robot should do based on what it perceives and its goals. Without effective decision-making systems, robots cannot adapt to changing conditions or handle complex tasks that require multiple steps or behavioral modes. Understanding decision-making approaches is essential for creating autonomous Physical AI systems.

## Core Concepts

### Rule-Based Systems

Rule-based systems make decisions using a set of "if-then" rules that map conditions to actions:

```
IF sensor_data.detects_obstacle_ahead THEN action_avoid_obstacle
IF battery_level < 10% THEN action_return_to_dock
```

For beginners: Think of rule-based systems like a recipe with simple conditional steps: "If the oil is smoking, turn down the heat. If the food is browned, flip it."

For intermediate learners: Rule-based systems use logical inference to determine actions from facts and rules. They can be expressed as production rules with condition-action pairs and processed by inference engines.

### Finite State Machines (FSMs)

FSMs model behavior as discrete states with transitions between them based on events or conditions:

1. **States**: Discrete behavioral modes (e.g., idle, moving, grasping)
2. **Transitions**: Conditions that cause state changes
3. **Actions**: Activities performed in each state

For beginners: An FSM is like a simple game where you can only be in one room at a time, and you move between rooms by going through doors that open under certain conditions.

For intermediate learners: FSMs have mathematical properties that make them predictable and analyzable. They can be deterministic (one transition per trigger) or non-deterministic (multiple possible transitions).

### Behavior Trees

Behavior trees organize decision-making as a tree of hierarchical nodes:

1. **Action Nodes**: Leaf nodes that perform actual behaviors
2. **Condition Nodes**: Check conditions in the environment
3. **Control Nodes**: Sequence, select, or modify execution flow
4. **Decorator Nodes**: Modify the behavior of child nodes

For beginners: Behavior trees are like decision trees but with more complex logic, allowing for sequences, fallbacks, and conditional execution.

For intermediate learners: Behavior trees offer modularity, reusability, and better error handling than FSMs. They're widely used in game AI and robotics for their flexibility and composability.

### Decision-Making Challenges

1. **Uncertainty**: Decisions must account for uncertain perception and outcomes
2. **Real-time Requirements**: Decisions must be made within time constraints
3. **Complexity**: Behavior must scale from simple to complex tasks
4. **Safety**: Decisions must prioritize safety of robot and environment
5. **Efficiency**: Decision-making should use minimal computational resources

For beginners: These challenges mean that simple decision rules might not work well in complex or unpredictable environments.

For intermediate learners: Advanced techniques like planning under uncertainty (POMDPs) or learning-based decision making can address these challenges more effectively than simple rule-based systems.

## Hands-on Section

### Implementing a Rule-Based Decision System

Let's create a simple rule-based system for a robot navigating and avoiding obstacles:

```python
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    MOVING_FORWARD = 2
    TURNING = 3
    AVOIDING_OBSTACLE = 4

class RuleBasedRobot:
    def __init__(self):
        self.state = RobotState.IDLE
        self.position = [0, 0]
        self.heading = 0  # Angle in radians
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.obstacle_angle = 0  # Relative to heading
        self.battery_level = 100.0
    
    def update_sensors(self, obstacle_detected, obstacle_distance, obstacle_angle):
        """Update sensor information"""
        self.obstacle_detected = obstacle_detected
        self.obstacle_distance = obstacle_distance
        self.obstacle_angle = obstacle_angle
        # Simulate battery drain
        self.battery_level -= 0.01
    
    def apply_rules(self):
        """Apply decision rules to determine next action"""
        # Rule 1: If battery is low, return to dock
        if self.battery_level < 15.0:
            return "RETURN_TO_DOCK"
        
        # Rule 2: If obstacle is very close, avoid it immediately
        if self.obstacle_detected and self.obstacle_distance < 0.5:
            return "AVOID_OBSTACLE_IMMEDIATE"
        
        # Rule 3: If obstacle is detected ahead, start avoidance
        if self.obstacle_detected and abs(self.obstacle_angle) < 0.5:  # Within 30 degrees
            return "AVOID_OBSTACLE"
        
        # Rule 4: If no obstacles ahead, move forward
        if not self.obstacle_detected or abs(self.obstacle_angle) > 0.5:
            return "MOVE_FORWARD"
        
        # Default: keep current behavior
        return "CONTINUE_CURRENT"

    def update_position(self, action, dt=0.1):
        """Update robot position based on action"""
        if action == "MOVE_FORWARD":
            self.state = RobotState.MOVING_FORWARD
            # Move forward
            self.position[0] += 0.5 * np.cos(self.heading) * dt
            self.position[1] += 0.5 * np.sin(self.heading) * dt
        
        elif action == "AVOID_OBSTACLE_IMMEDIATE":
            self.state = RobotState.AVOIDING_OBSTACLE
            # Turn away from obstacle
            self.heading += np.sign(self.obstacle_angle) * 1.0 * dt
            # Move slightly back
            self.position[0] -= 0.2 * np.cos(self.heading) * dt
            self.position[1] -= 0.2 * np.sin(self.heading) * dt
        
        elif action == "AVOID_OBSTACLE":
            self.state = RobotState.TURNING
            # Turn to avoid obstacle
            self.heading += np.sign(-self.obstacle_angle) * 0.5 * dt  # Turn away from obstacle
        
        elif action == "RETURN_TO_DOCK":
            self.state = RobotState.MOVING_FORWARD
            # Head towards origin (dock)
            target_angle = np.arctan2(-self.position[1], -self.position[0])
            self.heading = 0.7 * self.heading + 0.3 * target_angle  # Smooth turn
            self.position[0] += 0.4 * np.cos(self.heading) * dt
            self.position[1] += 0.4 * np.sin(self.heading) * dt
        
        elif action == "CONTINUE_CURRENT":
            # Continue current behavior
            if self.state == RobotState.MOVING_FORWARD:
                self.position[0] += 0.5 * np.cos(self.heading) * dt
                self.position[1] += 0.5 * np.sin(self.heading) * dt

# Simulate robot navigation
robot = RuleBasedRobot()
robot.position = [0, 0]
robot.heading = 0

# Simulation parameters
dt = 0.1
total_time = 20
time_points = np.arange(0, total_time, dt)

# Store robot path for visualization
path_x = [robot.position[0]]
path_y = [robot.position[1]]
battery_levels = [robot.battery_level]
headings = [robot.heading]

# Simulate a scenario with obstacles
for t in time_points[1:]:
    # Simulate sensor data
    # Create artificial obstacles at certain positions
    obstacle_detected = False
    obstacle_distance = float('inf')
    obstacle_angle = 0
    
    # Add some artificial obstacles
    if 5 < t < 7:  # Obstacle in path
        obstacle_detected = True
        obstacle_distance = 0.8
        obstacle_angle = 0.2  # Slightly to the right
    
    elif 12 < t < 14 and abs(robot.position[1]) < 2:  # Another obstacle
        obstacle_detected = True
        obstacle_distance = 0.6
        obstacle_angle = -0.3  # Slightly to the left
    
    # Update robot sensors
    robot.update_sensors(obstacle_detected, obstacle_distance, obstacle_angle)
    
    # Apply decision rules
    action = robot.apply_rules()
    
    # Update robot position
    robot.update_position(action, dt)
    
    # Store data
    path_x.append(robot.position[0])
    path_y.append(robot.position[1])
    battery_levels.append(robot.battery_level)
    headings.append(robot.heading)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(path_x, path_y, 'b-', linewidth=2, label='Robot Path')
plt.plot(0, 0, 'ro', markersize=10, label='Start/Dock')
plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
plt.title('Robot Navigation Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(time_points, battery_levels, 'r-', linewidth=2)
plt.title('Battery Level Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Battery %')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(time_points, [h * 180/np.pi for h in headings], 'g-', linewidth=2)
plt.title('Robot Heading Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heading (degrees)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(path_x, path_y, 'b-', linewidth=2)
plt.plot(0, 0, 'ro', markersize=10, label='Start/Dock')
# Show obstacles as red circles
obstacle_x = [x for i, x in enumerate(path_x) if 5 < time_points[i] < 7 or (12 < time_points[i] < 14 and abs(path_y[i]) < 2)]
obstacle_y = [y for i, y in enumerate(path_y) if 5 < time_points[i] < 7 or (12 < time_points[i] < 14 and abs(path_y[i]) < 2)]
plt.scatter(obstacle_x, obstacle_y, c='red', s=100, alpha=0.6, label='Detected Obstacles')
plt.title('Path with Obstacles')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Final position: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
print(f"Final battery: {robot.battery_level:.2f}%")
print(f"Final heading: {robot.heading * 180/np.pi:.2f} degrees")
```

### Implementing a Finite State Machine for Robot Behavior

Let's implement an FSM for a robot performing a search and rescue operation:

```python
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class SearchRobotState(Enum):
    IDLE = 1
    SEARCHING = 2
    EXAMINING_OBJECT = 3
    TRANSPORTING = 4
    RETURNING_TO_BASE = 5

class SearchRescueRobot:
    def __init__(self):
        self.state = SearchRobotState.IDLE
        self.position = [0, 0]
        self.carrying_item = False
        self.target_item_position = None
        self.base_position = [0, 0]
        self.items_found = 0
        self.battery_level = 100.0
        self.search_pattern = "spiral"
        
    def update_sensors(self, object_detected, object_position=None, object_type="unknown"):
        self.object_detected = object_detected
        self.object_position = object_position
        self.object_type = object_type
        
    def transition(self):
        """Handle state transitions based on conditions"""
        if self.battery_level < 10:
            self.state = SearchRobotState.RETURNING_TO_BASE
        elif self.state == SearchRobotState.IDLE:
            if self.object_detected and self.object_type == "item":
                self.state = SearchRobotState.EXAMINING_OBJECT
                self.target_item_position = self.object_position
            else:
                self.state = SearchRobotState.SEARCHING
        elif self.state == SearchRobotState.SEARCHING:
            if self.object_detected and self.object_type == "item":
                self.state = SearchRobotState.EXAMINING_OBJECT
                self.target_item_position = self.object_position
        elif self.state == SearchRobotState.EXAMINING_OBJECT:
            # After examining, if it's a target item, pick it up
            if self.object_type == "item" and np.linalg.norm(
                np.array(self.position) - np.array(self.target_item_position)
            ) < 0.5:
                self.carrying_item = True
                self.items_found += 1
                self.state = SearchRobotState.TRANSPORTING
        elif self.state == SearchRobotState.TRANSPORTING:
            # If at base with item, drop it off
            if (np.linalg.norm(np.array(self.position) - np.array(self.base_position)) < 0.5 
                and self.carrying_item):
                self.carrying_item = False
                self.state = SearchRobotState.SEARCHING
            elif self.battery_level < 30:  # Low battery while carrying item
                self.state = SearchRobotState.RETURNING_TO_BASE
        elif self.state == SearchRobotState.RETURNING_TO_BASE:
            # If at base and not carrying item, start searching again
            if (np.linalg.norm(np.array(self.position) - np.array(self.base_position)) < 0.5 
                and not self.carrying_item and self.battery_level > 80):
                self.state = SearchRobotState.SEARCHING
    
    def update_position(self, dt=0.1):
        """Update robot position based on current state"""
        if self.state == SearchRobotState.SEARCHING:
            # Spiral search pattern
            t = len(trajectory_x) * dt  # Time-based spiral
            radius = 0.2 * t
            angle = 0.5 * t
            self.position[0] = radius * np.cos(angle)
            self.position[1] = radius * np.sin(angle)
        
        elif self.state == SearchRobotState.EXAMINING_OBJECT:
            # Move toward target item
            if self.target_item_position:
                direction = np.array(self.target_item_position) - np.array(self.position)
                distance = np.linalg.norm(direction)
                if distance > 0.1:  # If not already at target
                    direction = direction / distance  # Normalize
                    self.position[0] += direction[0] * 0.5 * dt
                    self.position[1] += direction[1] * 0.5 * dt
        
        elif self.state == SearchRobotState.TRANSPORTING:
            # Move toward base
            direction = np.array(self.base_position) - np.array(self.position)
            distance = np.linalg.norm(direction)
            if distance > 0.1:
                direction = direction / distance  # Normalize
                self.position[0] += direction[0] * 0.6 * dt
                self.position[1] += direction[1] * 0.6 * dt
                
        elif self.state == SearchRobotState.RETURNING_TO_BASE:
            # Move toward base
            direction = np.array(self.base_position) - np.array(self.position)
            distance = np.linalg.norm(direction)
            if distance > 0.1:
                direction = direction / distance  # Normalize
                self.position[0] += direction[0] * 0.8 * dt  # Faster return when low battery
                self.position[1] += direction[1] * 0.8 * dt
        
        # Update battery based on activity
        if self.state == SearchRobotState.TRANSPORTING:
            self.battery_level -= 0.05  # Higher drain when carrying
        elif self.state == SearchRobotState.SEARCHING:
            self.battery_level -= 0.02  # Moderate drain during search
        else:
            self.battery_level -= 0.01  # Lower drain otherwise

# Simulate the search and rescue robot
robot = SearchRescueRobot()
robot.position = [0, 0]

dt = 0.1
total_time = 40
time_points = np.arange(0, total_time, dt)
trajectory_x = [robot.position[0]]
trajectory_y = [robot.position[1]]
battery_levels = [robot.battery_level]
states = [robot.state.value]
items_collected = [robot.items_found]

# Simulate with artificial "item" detections
for t in time_points[1:]:
    # Simulate object detection
    robot.object_detected = False
    robot.object_position = None
    
    # Place artificial items at specific locations
    if (abs(robot.position[0] - 2) < 0.3 and abs(robot.position[1] - 1) < 0.3 and 
        robot.state != SearchRobotState.TRANSPORTING):
        robot.object_detected = True
        robot.object_position = [2, 1]
        robot.object_type = "item"
    elif (abs(robot.position[0] + 1) < 0.3 and abs(robot.position[1] - 3) < 0.3 and 
          robot.state != SearchRobotState.TRANSPORTING):
        robot.object_detected = True
        robot.object_position = [-1, 3]
        robot.object_type = "item"
    
    # Handle state transitions
    robot.transition()
    
    # Update position based on current state
    robot.update_position(dt)
    
    # Store data
    trajectory_x.append(robot.position[0])
    trajectory_y.append(robot.position[1])
    battery_levels.append(robot.battery_level)
    states.append(robot.state.value)
    items_collected.append(robot.items_found)

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Robot Path')
plt.plot(0, 0, 'rs', markersize=10, label='Base Station')
# Mark collected items
item_positions = [[2, 1], [-1, 3]]  # Known item positions
for pos in item_positions:
    plt.plot(pos[0], pos[1], 'go', markersize=10, label=f'Item at {pos}')
plt.plot(trajectory_x[0], trajectory_y[0], 'bo', markersize=8, label='Start')
plt.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=8, label='End')
plt.title('Search and Rescue Robot Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(time_points, battery_levels, 'r-', linewidth=2)
plt.title('Battery Level Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Battery %')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(time_points, states, 'g-', linewidth=2, drawstyle='steps-post')
plt.title('Robot State Over Time')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.yticks([1, 2, 3, 4, 5], ['Idle', 'Searching', 'Examining', 'Transporting', 'Returning'])
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(time_points, items_collected, 'm-', linewidth=2)
plt.title('Items Collected Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Items Collected')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Final position: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
print(f"Items collected: {robot.items_found}")
print(f"Final battery: {robot.battery_level:.2f}%")
print(f"Final state: {robot.state.name}")
```

## Real-World Mapping

### Decision-Making Systems in Practice

Real-world robotic decision-making faces additional complexities:

- **Uncertainty Management**: Handling uncertain perception, actions, and environmental states
- **Multi-Objective Optimization**: Balancing competing goals (efficiency, safety, energy)
- **Coordination**: Managing decisions when multiple robots are present
- **Learning Adaptation**: Improving decision-making through experience

### Decision-Making System Properties

| Property | Requirement |
|----------|-------------|
| **Reactivity** | Respond quickly to environmental changes |
| **Robustness** | Continue operating despite failures |
| **Predictability** | Behave consistently under similar conditions |
| **Safety** | Prioritize safe behaviors and fail-safes |
| **Efficiency** | Achieve goals with minimal resource usage |

### Industrial Applications

- **Warehouse Robotics**: Deciding what items to pick, where to go, and how to navigate
- **Autonomous Vehicles**: Decision making for navigation, lane changes, and collision avoidance
- **Agricultural Robots**: Path planning, crop identification, and treatment decisions
- **Service Robots**: Task prioritization, obstacle handling, and human interaction

### Advanced Decision-Making Approaches

While rule-based systems, FSMs, and behavior trees are fundamental:

1. **Planning Algorithms**: Path planning, task planning, and motion planning
2. **Reinforcement Learning**: Learning decision policies through trial and error
3. **Classical AI Planning**: Symbolic planning using STRIPS, PDDL, or similar languages
4. **Hierarchical Task Networks**: Decomposing complex tasks into simpler subtasks

## Exercises

### Beginner Tasks
1. Run the rule-based robot simulation and observe how it responds to obstacles
2. Modify the rules to add new behaviors (e.g., avoid certain types of obstacles)
3. Run the FSM simulation to see state transitions during task execution
4. Change the robot's behavior parameters (speed, turning rate) in the simulation

### Stretch Challenges
1. Extend the FSM to include more states (e.g., charging, communication)
2. Implement a behavior tree for the same search-and-rescue task
3. Add uncertainty to the sensor model and see how it affects decision making

## Summary

This chapter explored fundamental decision-making approaches in Physical AI: rule-based systems, finite state machines, and behavior trees. We implemented examples of each approach showing how robots can make decisions based on their perceptions and goals.

Decision-making systems are crucial for autonomous Physical AI, determining what actions to take based on sensor inputs and objectives. Understanding these approaches provides the foundation for developing robots that can operate independently in complex environments.

In the next part of this book, we'll explore how Physical AI systems can learn and adapt, moving beyond pre-programmed behaviors.