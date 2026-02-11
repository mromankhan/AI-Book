---
sidebar_position: 2
---

# Chapter 3.2: Reinforcement Learning Basics

## Overview

In this chapter, you will learn:
- The fundamentals of reinforcement learning for robotics
- How agents learn through interaction with the environment
- The concept of simulated environments for safe learning
- Understanding the "reality gap" between simulation and real-world application

Reinforcement Learning (RL) enables robots to learn optimal behaviors through trial and error, making it particularly valuable for complex tasks that are difficult to engineer directly.

### Learning Objectives

By the end of this chapter, you will be able to:
1. Define the fundamental components of RL (agent, environment, states, actions, rewards)
2. Understand the exploration vs. exploitation dilemma in RL
3. Implement basic RL algorithms for simple robotic tasks
4. Explain the sim-to-real transfer problem in robotics

### Why This Matters

Reinforcement Learning is essential for Physical AI because it enables robots to learn complex behaviors that are difficult to program manually. RL allows robots to adapt to complex, changing environments and optimize their behavior through experience, making it crucial for autonomous systems that must operate in the real world.

## Core Concepts

### The RL Framework

Reinforcement Learning consists of:

1. **Agent**: The learning system (robot)
2. **Environment**: The world the agent interacts with
3. **State (s)**: Information about the current situation
4. **Action (a)**: What the agent can do
5. **Reward (r)**: Feedback about the quality of an action
6. **Policy (π)**: Strategy for selecting actions

For beginners: Think of RL like teaching someone to play a game by giving them points for good moves and deducting points for bad moves. The person learns which moves maximize points over time.

For intermediate learners: The RL problem is formalized as a Markov Decision Process (MDP) where the agent seeks to maximize cumulative reward over time.

### Agent, Environment, Reward Cycle

The RL process follows a continuous cycle:
1. Agent observes state (s) from environment
2. Agent selects action (a) based on its policy
3. Environment transitions to new state (s')
4. Environment provides reward (r) to agent
5. Cycle repeats, possibly with terminal state

For beginners: This is like playing a video game where you see the screen (state), decide what to do (action), the game updates (new state), and you get points or lose lives (reward).

For intermediate learners: The key is that the agent must learn a policy that maximizes long-term reward, not just immediate reward.

### Exploration vs. Exploitation

A fundamental RL challenge is balancing:
- **Exploration**: Trying new actions to discover better strategies
- **Exploitation**: Using known good actions to maximize immediate reward

For beginners: This is like choosing a restaurant - try a new place (exploration) or go to your favorite (exploitation).

For intermediate learners: Strategies like epsilon-greedy, UCB, and Thompson sampling address this tradeoff.

### Simulated Environments

RL agents typically train in simulated environments before deployment:

1. **Safety**: No risk of physical damage during learning
2. **Speed**: Simulations run faster than real-time
3. **Control**: Ability to reset and replay scenarios
4. **Cost**: No wear on physical hardware

For beginners: Simulation is like practicing driving in a virtual car before driving a real car - safer, cheaper, and faster to learn.

For intermediate learners: Simulations must balance simplicity (for fast training) with realism (for effective transfer).

### The Reality Gap

A major challenge in robotics is the "reality gap":
- Simulations are imperfect models of the real world
- Policies that work in simulation may fail in reality
- Differences in physics, sensor noise, and actuator dynamics

For beginners: The reality gap is like learning to drive on a video game but finding that real cars handle differently.

For intermediate learners: Techniques like domain randomization, system identification, and sim-to-real transfer algorithms address the reality gap.

## Hands-on Section

### Implementing Q-Learning for Grid Navigation

Let's implement a basic RL algorithm (Q-Learning) for a simple grid navigation task:

```python
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class GridEnvironment:
    """Simple 5x5 grid environment for navigation"""
    def __init__(self):
        self.size = 5
        self.state = (0, 0)  # Start at top-left
        self.goal = (4, 4)   # Goal at bottom-right
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # Fixed obstacles
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0), 'down': (1, 0),
            'left': (0, -1), 'right': (0, 1)
        }
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        action_idx = self.actions.index(action)
        
        # Calculate new position
        dr, dc = self.action_effects[action]
        new_r = max(0, min(self.size - 1, self.state[0] + dr))
        new_c = max(0, min(self.size - 1, self.state[1] + dc))
        
        # Check if new position is an obstacle
        if (new_r, new_c) in self.obstacles:
            # Stay in current position if obstacle
            new_r, new_c = self.state
        
        self.state = (new_r, new_c)
        
        # Calculate reward
        if self.state == self.goal:
            reward = 10  # Large positive reward for reaching goal
            done = True
        elif (new_r, new_c) in self.obstacles:
            reward = -1  # Small penalty for hitting obstacle
            done = False
        else:
            reward = -0.1  # Small time penalty to encourage efficiency
            done = False
        
        return self.state, reward, done, {}
    
    def get_state_features(self, state):
        """Convert state to features for learning algorithms"""
        # Simple features: coordinates and distance to goal
        r, c = state
        goal_r, goal_c = self.goal
        dist_to_goal = abs(r - goal_r) + abs(c - goal_c)
        return (r, c, dist_to_goal)

def epsilon_greedy_policy(Q, state, epsilon, actions):
    """Epsilon-greedy policy for exploration vs exploitation"""
    if random.random() < epsilon:
        # Explore: random action
        return random.choice(actions)
    else:
        # Exploit: best known action
        q_values = [Q.get((state, a), 0) for a in actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(max_actions)

# Q-Learning implementation
def q_learning(env, episodes=2000, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Q-Learning algorithm implementation"""
    Q = defaultdict(float)  # Q-value table
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(Q, state, epsilon, env.actions)
            
            # Take action and observe outcome
            next_state, reward, done, _ = env.step(action)
            
            # Q-Learning update
            best_next_action = max(env.actions, key=lambda a: Q.get((next_state, a), 0))
            td_target = reward + gamma * Q.get((next_state, best_next_action), 0)
            td_error = td_target - Q.get((state, action), 0)
            Q[(state, action)] += alpha * td_error
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    return Q, episode_rewards

# Train the Q-Learning agent
env = GridEnvironment()
Q_table, rewards = q_learning(env)

# Evaluate the learned policy
def evaluate_policy(env, Q, num_episodes=10):
    """Evaluate the learned policy"""
    results = []
    env = GridEnvironment()  # Fresh environment
    
    for episode in range(num_episodes):
        state = env.reset()
        path = [state]
        total_reward = 0
        done = False
        steps = 0
        max_steps = 50  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Use greedy policy (no exploration during evaluation)
            q_values = [Q.get((state, a), 0) for a in env.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(env.actions, q_values) if q == max_q]
            action = random.choice(best_actions)
            
            state, reward, done, _ = env.step(action)
            path.append(state)
            total_reward += reward
            steps += 1
        
        results.append({
            'path': path,
            'total_reward': total_reward,
            'reached_goal': done and state == env.goal,
            'steps': steps
        })
    
    return results

# Evaluate and display results
evaluation_results = evaluate_policy(env, Q_table)

print(f"Training completed over {len(rewards)} episodes")
print(f"Average reward in last 100 episodes: {np.mean(rewards[-100:]):.3f}")

successful_runs = [r for r in evaluation_results if r['reached_goal']]
print(f"Successful runs: {len(successful_runs)}/{len(evaluation_results)}")

# Plot learning curve
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards, alpha=0.7)
plt.title('Learning Curve: Episode Reward Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True, alpha=0.3)

# Plot rewards for last 100 episodes moving average
plt.subplot(1, 3, 2)
moving_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
plt.plot(moving_avg)
plt.title('Moving Average Reward (Last 100 Episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True, alpha=0.3)

# Visualize a successful path
plt.subplot(1, 3, 3)
if successful_runs:
    best_run = min(successful_runs, key=lambda r: r['steps'])
    path = best_run['path']
    path_r = [pos[0] for pos in path]
    path_c = [pos[1] for pos in path]
    
    plt.plot(path_c, path_r, 'b-', linewidth=3, label='Agent Path', zorder=5)
    plt.plot(path_c[0], path_r[0], 'go', markersize=15, label='Start', zorder=5)
    plt.plot(path_c[-1], path_r[-1], 'ro', markersize=15, label='Goal', zorder=5)
    
    # Plot obstacles
    obs_r, obs_c = zip(*env.obstacles)
    plt.scatter(obs_c, obs_r, c='red', s=200, marker='s', label='Obstacles', zorder=5)
    
    plt.title(f'Successful Path (Steps: {best_run["steps"]})')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()  # Grid origin is top-left
    
    # Set axis to show grid
    plt.xticks(range(env.size))
    plt.yticks(range(env.size))
    plt.xlim(-0.5, env.size-0.5)
    plt.ylim(-0.5, env.size-0.5)

plt.tight_layout()
plt.show()
```

### Simulated Environment with Sim-to-Real Transfer Example

Let's create a more complex example that demonstrates the sim-to-real transfer challenge:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class SimulatedRobotEnv:
    """Simulated 2D navigation environment"""
    def __init__(self, real_dynamics=False):
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.target = np.array([5.0, 5.0])
        self.real_dynamics = real_dynamics  # Toggle realistic physics
        self.time_step = 0.1
        self.max_force = 10.0
        self.actions = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right', 'none']
        
    def reset(self):
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        # Random target for more variety
        self.target = np.random.uniform(3.0, 7.0, size=2)
        return np.concatenate([self.position, self.velocity, self.target])
    
    def step(self, action_idx):
        # Convert action index to force vector
        forces = {
            0: [0, 1],      # up
            1: [0, -1],     # down
            2: [-1, 0],     # left
            3: [1, 0],      # right
            4: [-1, 1],     # up-left
            5: [1, 1],      # up-right
            6: [-1, -1],    # down-left
            7: [1, -1],     # down-right
            8: [0, 0]       # none
        }
        
        force = np.array(forces[action_idx]) * self.max_force
        
        # Apply dynamics (physics simulation)
        if self.real_dynamics:
            # More realistic dynamics with drag
            acceleration = force - 0.5 * self.velocity - 0.1 * self.velocity * np.linalg.norm(self.velocity)
            acceleration = acceleration / 1.0  # mass = 1
        else:
            # Simplified dynamics for easier learning
            acceleration = force - 0.1 * self.velocity  # Simple damping
            acceleration = acceleration / 1.0  # mass = 1
        
        # Update state (simple Euler integration)
        self.velocity += acceleration * self.time_step
        self.position += self.velocity * self.time_step
        
        # Calculate reward
        distance_to_target = np.linalg.norm(self.position - self.target)
        
        # Dense reward based on distance (closer is better)
        reward = -distance_to_target * 0.1  # Scale down for stability
        
        # Bonus for getting close
        if distance_to_target < 0.5:
            reward += 1.0
        
        # Bonus for reaching target
        done = False
        if distance_to_target < 0.2:
            reward += 10.0
            done = True
        
        # Small penalty for large actions to encourage efficiency
        action_magnitude = np.linalg.norm(force)
        reward -= action_magnitude * 0.001
        
        # State includes position, velocity, and target
        state = np.concatenate([self.position, self.velocity, self.target])
        
        return state, reward, done, {}

def simple_navigation_agent(observation):
    """Simple rule-based agent for comparison"""
    pos = observation[:2]
    vel = observation[2:4]
    target = observation[4:6]
    
    # Simple proportional controller
    error = target - pos
    desired_vel = error * 0.5  # Proportional gain
    vel_error = desired_vel - vel
    
    # Convert velocity error to force
    force = vel_error * 2.0  # Assume mass of 2
    
    # Choose action based on force direction
    force_norm = np.linalg.norm(force)
    if force_norm < 0.1:  # Very small force
        return 8  # No action
    
    # Determine action based on force direction
    force_angle = np.arctan2(force[1], force[0])
    angle_deg = np.degrees(force_angle) % 360
    
    if 337.5 <= angle_deg < 360 or 0 <= angle_deg < 22.5:
        return 3  # right
    elif 22.5 <= angle_deg < 67.5:
        return 5  # up-right
    elif 67.5 <= angle_deg < 112.5:
        return 0  # up
    elif 112.5 <= angle_deg < 157.5:
        return 4  # up-left
    elif 157.5 <= angle_deg < 202.5:
        return 2  # left
    elif 202.5 <= angle_deg < 247.5:
        return 6  # down-left
    elif 247.5 <= angle_deg < 292.5:
        return 1  # down
    else:
        return 7  # down-right

# Test both simulated environments
def run_episode(env, agent_func=None, max_steps=200):
    """Run an episode with either learned policy or random actions"""
    state = env.reset()
    total_reward = 0
    trajectory = [env.position.copy()]
    
    for step in range(max_steps):
        if agent_func:
            # Use learned agent
            action = agent_func(state)
        else:
            # Random action for comparison
            action = np.random.randint(0, 9)
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        trajectory.append(env.position.copy())
        
        if done:
            break
    
    return total_reward, env.position.copy(), np.array(trajectory), done

# Compare simulation environments
sim_env_simple = SimulatedRobotEnv(real_dynamics=False)
sim_env_realistic = SimulatedRobotEnv(real_dynamics=True)

print("Testing navigation with simple dynamics:")
rewards_simple = []
traj_simple = []
for episode in range(10):
    reward, final_pos, traj, done = run_episode(sim_env_simple, simple_navigation_agent)
    rewards_simple.append(reward)
    traj_simple.append(traj)

print(f"Average reward (simple): {np.mean(rewards_simple):.2f}")
print(f"Success rate: {sum(1 for r in rewards_simple if r > 5)}/{len(rewards_simple)}")

print("\nTesting navigation with realistic dynamics:")
rewards_realistic = []
traj_realistic = []
for episode in range(10):
    reward, final_pos, traj, done = run_episode(sim_env_realistic, simple_navigation_agent)
    rewards_realistic.append(reward)
    traj_realistic.append(traj)

print(f"Average reward (realistic): {np.mean(rewards_realistic):.2f}")
print(f"Success rate: {sum(1 for r in rewards_realistic if r > 5)}/{len(rewards_realistic)}")

# Visualize the difference
plt.figure(figsize=(15, 10))

# Plot trajectories in simple environment
plt.subplot(2, 2, 1)
for i, traj in enumerate(traj_simple[:5]):  # Plot first 5 trajectories
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=f'Trajectory {i+1}' if i < 3 else "")
plt.title('Simple Dynamics Environment')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot trajectories in realistic environment
plt.subplot(2, 2, 2)
for i, traj in enumerate(traj_realistic[:5]):  # Plot first 5 trajectories
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, label=f'Trajectory {i+1}' if i < 3 else "")
plt.title('Realistic Dynamics Environment')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare reward distributions
plt.subplot(2, 2, 3)
plt.hist(rewards_simple, alpha=0.7, label='Simple Dynamics', bins=10)
plt.hist(rewards_realistic, alpha=0.7, label='Realistic Dynamics', bins=10)
plt.title('Reward Distribution Comparison')
plt.xlabel('Total Reward')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare success rates
plt.subplot(2, 2, 4)
success_simple = sum(1 for r in rewards_simple if r > 5)
success_realistic = sum(1 for r in rewards_realistic if r > 5)
success_rates = [success_simple, success_realistic]
labels = ['Simple Dynamics', 'Realistic Dynamics']
plt.bar(labels, success_rates, color=['blue', 'red'], alpha=0.7)
plt.title('Success Rate Comparison (Reward > 5)')
plt.ylabel('Number of Successful Episodes')
for i, v in enumerate(success_rates):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\nReality Gap Example:")
print(f"Success rate in simple simulation: {success_simple}/10")
print(f"Success rate in realistic simulation: {success_realistic}/10")
print(f"Difference: {success_simple - success_realistic}")
```

## Real-World Mapping

### RL in Real Robotics Applications

Real-world RL deployment has unique challenges:

- **Safety Constraints**: RL agents must learn without causing damage
- **Sample Efficiency**: Real robots have limited time for training
- **Embodied Cognition**: Physical embodiment affects learning process
- **Transfer Learning**: Adapting from simulation to reality
- **Continuous Adaptation**: Environments change over time

### RL Applications in Robotics

| Domain | RL Application | Benefits |
|--------|----------------|----------|
| Manipulation | Learning grasping and manipulation skills | Handles object variations |
| Navigation | Learning to navigate in complex environments | Adapts to dynamic obstacles |
| Control | Learning complex motor control patterns | Optimizes for efficiency |
| Human Interaction | Learning to work with humans | Adapts to human preferences |

### Sim-to-Real Transfer Techniques

| Technique | Description |
|-----------|-------------|
| **Domain Randomization** | Training with randomized simulation parameters |
| **System Identification** | Modeling real robot to tune simulation |
| **Learning from Demonstration** | Bootstrapping with human examples |
| **Adaptive Control** | Adjusting on real robot during deployment |

### Advanced RL Approaches

Beyond basic Q-learning:

1. **Deep Q-Networks (DQN)**: Using neural networks as function approximators
2. **Policy Gradient Methods**: Directly optimizing policy parameters
3. **Actor-Critic Methods**: Combining value-based and policy-based approaches
4. **Model-Based RL**: Learning environment dynamics for planning

## Exercises

### Beginner Tasks
1. Run the Q-Learning example and observe how the agent learns to navigate
2. Change the epsilon value to see how exploration rate affects learning
3. Modify the reward function and observe the impact on agent behavior
4. Run the sim-to-real example to see the reality gap in action

### Stretch Challenges
1. Implement a Deep Q-Network for the navigation task
2. Create a simulation with additional complexities (moving obstacles, partial observability)
3. Design a transfer learning experiment from one simulation to another

## Summary

This chapter covered the fundamentals of reinforcement learning for robotics, including the core concepts of agents, environments, and the exploration-exploitation tradeoff. We implemented a Q-Learning algorithm for navigation and demonstrated the sim-to-real transfer challenge.

Reinforcement Learning offers powerful approaches for learning complex behaviors in Physical AI systems. However, the sim-to-real gap remains a significant challenge that requires careful consideration in real-world deployments.

In the next part of this book, we'll explore humanoid-specific robotics challenges and concepts.