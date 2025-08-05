"""
Custom TD3 implementation for Stable Baselines 3 with Constrained Reinforcement Learning (CRL):
1. Huber Loss for the critic for more robust training.
2. Adaptive Lagrange multiplier λ for smoothness constraints (instead of fixed penalty).
3. Separate learning rate schedules for the actor and critic.
4. Constraint-based penalty that adapts based on actual policy smoothness.
"""

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy


class CustomTD3_AC(TD3):
    """
    Custom TD3 Agent with Constrained Reinforcement Learning (CRL) using adaptive Lagrange multipliers.
    
    Instead of a fixed penalty coefficient, this implementation uses a learnable Lagrange multiplier λ
    that adapts based on constraint violations. This provides principled, adaptive smoothness control.
    
    Key CRL Features:
    - Adaptive λ (lambda) parameter that increases when policy is too erratic
    - Constraint threshold defining maximum allowed action change
    - Automatic penalty adjustment based on measured constraint violations
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        actor_learning_rate: Union[float, Schedule],
        critic_learning_rate: Union[float, Schedule],
        # CRL-specific parameters
        constraint_threshold: float = 0.05,  # Maximum allowed average squared action change
        lambda_lr: float = 1e-3,  # Learning rate for Lagrange multiplier λ
        initial_lambda: float = 0.1,  # Initial value for λ
        # Existing parameters
        action_reg_coef: float = 0.1,  # Fallback for non-CRL mode
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_grad_norm: float = 1.0,
        overall_total_timesteps: Optional[int] = None,
        use_crl: bool = True,  # Enable/disable CRL mode
    ):
        # We need to call the parent init, but we'll manage the learning rate ourselves.
        # Pass a dummy value for learning_rate to the parent.
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=0.0,  # Dummy value, will be overridden
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,  # We'll set up the model ourselves
        )

        self.max_grad_norm = max_grad_norm
        self.action_reg_coef = action_reg_coef  # Fallback for non-CRL mode
        self.overall_total_timesteps = overall_total_timesteps

        # CRL-specific parameters
        self.use_crl = use_crl
        self.constraint_threshold = constraint_threshold  # Maximum allowed E[C_t]
        self.lambda_lr = lambda_lr
        
        # Store the separate learning rate schedules
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        # Initialize Lagrange multiplier λ as a learnable parameter
        if self.use_crl:
            self.lambda_param = torch.tensor(initial_lambda, device=self.device, requires_grad=False)  # No gradients needed
            
            # Track constraint violations for monitoring
            self.constraint_history = []
            self.lambda_history = []
            
            if self.verbose > 0:
                print(f"🎯 CRL Mode Enabled:")
                print(f"   Constraint threshold (d): {constraint_threshold:.4f}")
                print(f"   Initial λ: {initial_lambda:.4f}")
                print(f"   λ learning rate: {lambda_lr:.4f}")
        else:
            if self.verbose > 0:
                print(f"🚫 CRL Mode Disabled - Using fixed penalty: {action_reg_coef:.4f}")
        
        if _init_setup_model:
            self._setup_model()
            self._setup_custom_optimizers()

    def _create_schedule(self, lr: Union[float, Schedule]) -> Schedule:
        """Helper to create a schedule from a float."""
        if isinstance(lr, (float, int)):
            return lambda progress: lr
        return lr

    def _setup_custom_optimizers(self) -> None:
        """Create actor, critic, and lambda optimizers with separate learning rates."""
        self.actor_lr_schedule = self._create_schedule(self.actor_learning_rate)
        self.critic_lr_schedule = self._create_schedule(self.critic_learning_rate)
        
        # Assume Adam optimizer if not specified
        optimizer_class = self.policy_kwargs.get("optimizer_class", torch.optim.Adam)

        # Create optimizers with the initial learning rate
        self.actor.optimizer = optimizer_class(
            self.actor.parameters(), lr=self.actor_lr_schedule(1.0)
        )
        self.critic.optimizer = optimizer_class(
            self.critic.parameters(), lr=self.critic_lr_schedule(1.0)
        )
        
        # Note: No optimizer needed for λ - we use direct updates

    def _update_learning_rate(self, optimizers) -> None:
        """
        Update learning rates according to schedules.
        This is called automatically by the training loop.
        """
        if self.overall_total_timesteps is not None:
            # When using a custom training loop, progress must be calculated manually.
            progress = 1.0 - self.num_timesteps / self.overall_total_timesteps
        else:
            # Default SB3 behavior.
            progress = self._current_progress_remaining
        
        # Update actor learning rate
        new_actor_lr = self.actor_lr_schedule(progress)
        for param_group in self.actor.optimizer.param_groups:
            param_group["lr"] = new_actor_lr
        
        # Update critic learning rate
        new_critic_lr = self.critic_lr_schedule(progress)
        for param_group in self.critic.optimizer.param_groups:
            param_group["lr"] = new_critic_lr
            
        # Log the learning rates for monitoring
        self.logger.record("train/actor_lr", new_actor_lr)
        self.logger.record("train/critic_lr", new_critic_lr)
        
        # Log CRL-specific metrics
        if self.use_crl:
            current_lambda = self.lambda_param.item()
            self.logger.record("train/lambda", current_lambda)
            if len(self.constraint_history) > 0:
                self.logger.record("train/constraint_violation", self.constraint_history[-1])
                self.logger.record("train/constraint_threshold", self.constraint_threshold)

    def _update_lambda(self, constraint_cost: float) -> None:
        """
        Update Lagrange multiplier λ based on constraint violation.
        
        λ_new = max(0, λ_old + α * (E[C_t] - d))
        
        Args:
            constraint_cost: Current measured constraint cost E[C_t]
        """
        if not self.use_crl:
            return
            
        # Calculate constraint violation
        constraint_violation = constraint_cost - self.constraint_threshold
        
        # 💡 GRADIENT FIX: Invert the gradient for lambda update
        # The standard update is for minimization, but we want to *increase* lambda
        # when the constraint is violated (cost > threshold).
        # We achieve this by negating the violation term.
        # λ_new = max(0, λ_old - α * (threshold - constraint_cost))
        with torch.no_grad():
            # Get current lambda value
            current_lambda = self.lambda_param.item()
            
            # Apply the corrected CRL update rule (gradient is inverted)
            gradient = constraint_violation  # Standard gradient
            new_lambda = max(0.0, current_lambda + self.lambda_lr * gradient)
            
            # Update the parameter
            self.lambda_param.fill_(new_lambda)
        
        # Track for monitoring
        self.constraint_history.append(constraint_cost)
        self.lambda_history.append(self.lambda_param.item())
        
        # Keep only recent history for efficiency
        if len(self.constraint_history) > 1000:
            self.constraint_history = self.constraint_history[-500:]
            self.lambda_history = self.lambda_history[-500:]
        
        if self.verbose > 1 and self._n_updates % 100 == 0:  # Log every 100 updates
            print(f"🎯 CRL Update: E[C_t]={constraint_cost:.4f}, d={self.constraint_threshold:.4f}, "
                  f"violation={constraint_violation:.4f}, λ={self.lambda_param.item():.4f}")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Override the default training loop to use:
        1. Huber loss for the critic
        2. Adaptive Lagrange multiplier for action smoothness (CRL)
        3. Custom learning rate updates
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        constraint_costs = []  # Track constraint violations for CRL
        
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Handle both dict and tensor observations
                if isinstance(replay_data.next_observations, dict):
                    batch_size_ = replay_data.next_observations['observations'].shape[0]
                else:
                    batch_size_ = replay_data.next_observations.shape[0]
                action_dim = self.action_space.shape[0]
                device = self.device

                noise = torch.normal(
                    mean=0.0,
                    std=self.target_policy_noise,
                    size=(batch_size_, action_dim),
                    device=device
                ).clamp(-self.target_noise_clip, self.target_noise_clip)

                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                target_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                target_q_values, _ = torch.min(target_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_values

            # ✅ CRITIC UPDATE: Use Huber loss for robustness
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.smooth_l1_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            # ✅ ACTOR UPDATE: Apply CRL or fixed penalty
            if self._n_updates % self.policy_delay == 0:
                obs = replay_data.observations
                a_cur = self.actor(obs)
                actor_loss = -self.critic.q1_forward(obs, a_cur).mean()

                # Calculate action smoothness constraint
                with torch.no_grad():
                    a_next = self.actor(replay_data.next_observations)
                
                # Compute constraint cost: E[(a_next - a_cur)^2]
                action_change_squared = (a_next - a_cur).pow(2).sum(dim=1).mean()
                constraint_cost = action_change_squared.item()
                constraint_costs.append(constraint_cost)
                
                if self.use_crl:
                    # ✅ CRL MODE: Use adaptive Lagrange multiplier
                    current_lambda = torch.clamp(self.lambda_param, min=0.0)  # Ensure non-negative
                    smooth_loss = current_lambda * action_change_squared
                    
                    # Update λ based on constraint violation
                    self._update_lambda(constraint_cost)
                    
                else:
                    # 🚫 FIXED PENALTY MODE: Use fixed coefficient
                    smooth_loss = self.action_reg_coef * action_change_squared
                
                actor_loss = actor_loss + smooth_loss
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        # Logging
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        
        # CRL-specific logging
        if self.use_crl and len(constraint_costs) > 0:
            avg_constraint_cost = np.mean(constraint_costs)
            self.logger.record("train/avg_constraint_cost", avg_constraint_cost)
            self.logger.record("train/constraint_violation", avg_constraint_cost - self.constraint_threshold)

    def get_crl_stats(self) -> Dict[str, float]:
        """Get CRL-specific statistics for monitoring."""
        if not self.use_crl:
            return {"crl_enabled": False}
        
        stats = {
            "crl_enabled": True,
            "current_lambda": self.lambda_param.item(),
            "constraint_threshold": self.constraint_threshold,
        }
        
        if len(self.constraint_history) > 0:
            recent_violations = self.constraint_history[-10:]  # Last 10 measurements
            stats.update({
                "avg_recent_constraint_cost": np.mean(recent_violations),
                "avg_recent_violation": np.mean(recent_violations) - self.constraint_threshold,
                "num_measurements": len(self.constraint_history)
            })
        
        return stats 