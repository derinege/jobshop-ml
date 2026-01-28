"""
Reinforcement Learning Environment (Skeleton)
Provides a step-by-step scheduling simulation for RL training.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from data_loader import SchedulingInstance
from graph_builder import GraphBuilder, SchedulingGraph
import config


class JobShopEnv:
    """
    Gym-style environment for job shop scheduling.
    Allows step-by-step scheduling decisions with RL agents.
    
    This is a skeleton implementation that can be extended for full RL training.
    """
    
    def __init__(self, 
                 instance: SchedulingInstance,
                 weights: Dict[str, float] = None,
                 graph_builder: GraphBuilder = None):
        """
        Initialize environment.
        
        Args:
            instance: SchedulingInstance to schedule
            weights: Objective weights for reward calculation
            graph_builder: GraphBuilder for state representation
        """
        self.instance = instance
        self.weights = weights if weights else config.WEIGHTS
        self.graph_builder = graph_builder if graph_builder else GraphBuilder()
        
        # Environment state
        self.current_time = 0
        self.scheduled_ops = {}  # (job, op) -> start_time
        self.machine_available_times = {mc: 0 for mc in config.MACHINE_CLASSES}
        self.done = False
        
        # Track metrics for reward
        self.completion_times = {}  # (job, op) -> completion_time
        self.waiting_times = []
        self.idle_times = []
    
    def reset(self) -> SchedulingGraph:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state graph
        """
        self.current_time = 0
        self.scheduled_ops = {}
        self.machine_available_times = {mc: 0 for mc in config.MACHINE_CLASSES}
        self.done = False
        self.completion_times = {}
        self.waiting_times = []
        self.idle_times = []
        
        # Return initial state
        state = self.graph_builder.build_dynamic_graph(
            self.instance, 
            self.scheduled_ops, 
            self.current_time
        )
        
        return state
    
    def get_available_operations(self) -> List[Tuple[str, str]]:
        """
        Get list of operations that can be scheduled now.
        
        Returns:
            List of (job, op_name) tuples
        """
        available = []
        
        for (job, op_name), data in self.instance.operations.items():
            # Already scheduled?
            if (job, op_name) in self.scheduled_ops:
                continue
            
            # Check precedence constraints
            can_schedule = True
            for (j, op_a, op_b) in self.instance.precedence:
                if j == job and op_b == op_name:
                    # op_a must be complete
                    if (j, op_a) not in self.completion_times:
                        can_schedule = False
                        break
                    
                    if self.completion_times[(j, op_a)] > self.current_time:
                        can_schedule = False
                        break
            
            if can_schedule:
                available.append((job, op_name))
        
        return available
    
    def step(self, action: Tuple[str, str]) -> Tuple[SchedulingGraph, float, bool, Dict]:
        """
        Execute one scheduling step.
        
        Args:
            action: (job, op_name) tuple representing operation to schedule
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        
        job, op_name = action
        
        # Validate action
        if (job, op_name) not in self.instance.operations:
            raise ValueError(f"Invalid operation: {action}")
        
        if (job, op_name) in self.scheduled_ops:
            raise ValueError(f"Operation already scheduled: {action}")
        
        # Get operation data
        op_data = self.instance.operations[(job, op_name)]
        machine_class = op_data['machine_class']
        duration = op_data['duration']
        
        # Compute start time (max of current time and machine availability)
        earliest_start = max(self.current_time, 
                            self.machine_available_times[machine_class])
        
        # Check precedence constraints for actual start time
        for (j, op_a, op_b) in self.instance.precedence:
            if j == job and op_b == op_name:
                if (j, op_a) in self.completion_times:
                    earliest_start = max(earliest_start, 
                                        self.completion_times[(j, op_a)])
        
        start_time = earliest_start
        completion_time = start_time + duration
        
        # Schedule the operation
        self.scheduled_ops[(job, op_name)] = start_time
        self.completion_times[(job, op_name)] = completion_time
        self.machine_available_times[machine_class] = completion_time
        
        # Track waiting time (if operation couldn't start immediately)
        if start_time > self.current_time:
            self.waiting_times.append(start_time - self.current_time)
        
        # Track idle time on machine
        if start_time > self.machine_available_times[machine_class]:
            self.idle_times.append(start_time - self.machine_available_times[machine_class])
        
        # Update current time to completion of this operation
        self.current_time = completion_time
        
        # Check if done
        self.done = len(self.scheduled_ops) == len(self.instance.operations)
        
        # Compute reward (only at the end for episodic tasks)
        if self.done:
            reward = self._compute_final_reward()
        else:
            reward = 0.0  # Sparse reward
        
        # Next state
        next_state = self.graph_builder.build_dynamic_graph(
            self.instance,
            self.scheduled_ops,
            self.current_time
        )
        
        # Info
        info = {
            'scheduled_ops': len(self.scheduled_ops),
            'total_ops': len(self.instance.operations),
            'current_time': self.current_time
        }
        
        return next_state, reward, self.done, info
    
    def _compute_final_reward(self) -> float:
        """
        Compute final reward based on weighted objective (3 components).
        
        Returns:
            Negative weighted objective (we want to minimize)
        """
        # Makespan
        makespan = max(self.completion_times.values()) if self.completion_times else 0
        
        # Waiting times (already tracked)
        total_waiting = sum(self.waiting_times)
        
        # Idle times (already tracked)
        total_idle = sum(self.idle_times)
        
        # Weighted objective (3 components)
        objective = (
            self.weights['w_MS'] * makespan +
            self.weights['w_W'] * total_waiting +
            self.weights['w_G'] * total_idle
        )
        
        # Return negative (since we want to minimize, but RL maximizes reward)
        return -objective


class RLTrainer:
    """
    Skeleton for RL training with PPO-style algorithm.
    
    TODO: Implement full PPO training loop with:
    - Experience collection (rollouts)
    - Advantage estimation (GAE)
    - Policy and value updates
    - Entropy regularization
    
    This is provided as a starting point for future RL development.
    """
    
    def __init__(self,
                 model,
                 env_creator,
                 config_dict: Dict = None):
        """
        Initialize RL trainer.
        
        Args:
            model: GNN policy model
            env_creator: Function that creates new environments
            config_dict: RL configuration
        """
        self.model = model
        self.env_creator = env_creator
        self.config = config_dict if config_dict else config.RL_CONFIG
        
        # TODO: Initialize optimizer, value network, etc.
    
    def collect_rollouts(self, num_episodes: int) -> List[Dict]:
        """
        Collect rollout data from multiple episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of episode data
        
        TODO: Implement rollout collection
        """
        rollouts = []
        
        for _ in range(num_episodes):
            env = self.env_creator()
            episode_data = self._run_episode(env)
            rollouts.append(episode_data)
        
        return rollouts
    
    def _run_episode(self, env: JobShopEnv) -> Dict:
        """
        Run one episode in the environment.
        
        Returns:
            Episode data (states, actions, rewards, etc.)
            
        TODO: Implement episode execution
        """
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        done = False
        
        while not done:
            # TODO: Use policy to select action
            available_ops = env.get_available_operations()
            
            if not available_ops:
                break
            
            # Random action for now (placeholder)
            action = available_ops[0]
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': sum(rewards)
        }
    
    def train(self, num_iterations: int):
        """
        Train the policy with PPO.
        
        Args:
            num_iterations: Number of training iterations
            
        TODO: Implement full PPO training loop
        """
        print("RL training not fully implemented yet.")
        print("This is a skeleton for future development.")
        print("Key steps to implement:")
        print("  1. Collect rollouts with current policy")
        print("  2. Compute advantages with GAE")
        print("  3. Update policy with PPO objective")
        print("  4. Update value function")
        print("  5. Track and log metrics")
        
        # Placeholder training loop
        for iteration in range(num_iterations):
            # TODO: Implement training iteration
            pass


if __name__ == "__main__":
    print("=== RL Environment Test ===\n")
    
    from data_loader import DataLoader
    
    # Create a small instance
    loader = DataLoader()
    loader.load_data()
    instances = loader.generate_random_instances(n_instances=1, min_jobs=2, max_jobs=3)
    
    if instances:
        instance = instances[0]
        print(f"Instance: {instance}")
        
        # Create environment
        env = JobShopEnv(instance)
        
        # Test environment
        print("\nTesting environment:")
        state = env.reset()
        print(f"Initial state: {state.num_nodes} nodes")
        
        episode_reward = 0
        step = 0
        
        while not env.done:
            available = env.get_available_operations()
            print(f"\nStep {step}: {len(available)} operations available")
            
            if not available:
                print("No operations available!")
                break
            
            # Take random action
            action = available[0]
            print(f"  Scheduling: {action}")
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            print(f"  Reward: {reward:.2f}, Done: {done}")
            print(f"  Progress: {info['scheduled_ops']}/{info['total_ops']}")
            
            step += 1
            
            if step > 20:  # Safety limit
                break
        
        print(f"\nEpisode completed!")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Makespan: {max(env.completion_times.values()):.2f}")
