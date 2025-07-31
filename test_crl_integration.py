#!/usr/bin/env python3
"""
Test script to validate Constrained Reinforcement Learning (CRL) integration.
This script tests the CRL functionality without running a full training session.
"""

import sys
import numpy as np
import torch
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import get_linear_fn

# Import our modules
import config
from enviorment import PortfolioEnv
from attention_policy import SimpleMlpTD3Policy
from custom_td3_actor_critic import CustomTD3_AC


def test_crl_config():
    """Test that CRL configuration loading works correctly."""
    print("🧪 Testing CRL configuration loading...")
    
    # Test all CRL profiles
    for profile_name in ["conservative", "balanced", "aggressive", "adaptive"]:
        print(f"\n📋 Testing profile: {profile_name}")
        crl_config = config.get_crl_config(profile_name)
        
        required_keys = ["use_crl", "constraint_threshold", "lambda_lr", "initial_lambda", "action_reg_coef"]
        for key in required_keys:
            assert key in crl_config, f"Missing key '{key}' in CRL config for profile '{profile_name}'"
            print(f"   ✅ {key}: {crl_config[key]}")
    
    # Test invalid profile (should fallback to balanced)
    print(f"\n📋 Testing invalid profile (should fallback to 'balanced'):")
    crl_config = config.get_crl_config("invalid_profile")
    assert crl_config["constraint_threshold"] == config.CRL_PROFILES["balanced"]["constraint_threshold"]
    print(f"   ✅ Fallback works correctly")
    
    print("✅ CRL configuration tests passed!")


def test_crl_model_creation():
    """Test that CRL-enabled TD3 model can be created successfully."""
    print("\n🧪 Testing CRL model creation...")
    
    # Create environment
    env = PortfolioEnv(reward_type="TRANSACTION_COST")
    
    # Test model creation with CRL enabled
    for profile_name in ["conservative", "balanced", "aggressive"]:
        print(f"\n📋 Testing model creation with profile: {profile_name}")
        
        crl_config = config.get_crl_config(profile_name)
        
        # Create learning rate schedules
        actor_lr_schedule = get_linear_fn(start=1e-4, end=5e-5, end_fraction=0.9)
        critic_lr_schedule = get_linear_fn(start=5e-4, end=1e-4, end_fraction=0.9)
        
        # Create action noise
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=0.2 * np.ones(env.action_space.shape[-1]),
            theta=0.15
        )
        
        # Policy kwargs
        policy_kwargs = {
            "net_arch": {"pi": [400, 300], "qf": [400, 300]},
            "actor_lr": 1e-4,
            "critic_lr": 1e-3
        }
        
        try:
            model = CustomTD3_AC(
                SimpleMlpTD3Policy,
                env,
                verbose=0,  # Reduce verbosity for testing
                actor_learning_rate=actor_lr_schedule,
                critic_learning_rate=critic_lr_schedule,
                policy_kwargs=policy_kwargs,
                batch_size=64,  # Smaller batch for testing
                buffer_size=1000,  # Smaller buffer for testing
                learning_starts=10,  # Quick start for testing
                gamma=0.05,
                tau=0.005,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                action_noise=action_noise,
                device="auto",
                overall_total_timesteps=1000,  # Small number for testing
                # CRL parameters
                use_crl=crl_config['use_crl'],
                constraint_threshold=crl_config['constraint_threshold'],
                lambda_lr=crl_config['lambda_lr'],
                initial_lambda=crl_config['initial_lambda'],
                action_reg_coef=crl_config['action_reg_coef'],
            )
            
            print(f"   ✅ Model created successfully")
            
            # Test CRL stats retrieval
            crl_stats = model.get_crl_stats()
            print(f"   📊 CRL enabled: {crl_stats['crl_enabled']}")
            if crl_stats['crl_enabled']:
                print(f"   📊 Initial λ: {crl_stats['current_lambda']:.4f}")
                print(f"   📊 Constraint threshold: {crl_stats['constraint_threshold']:.4f}")
            
            # Test that the model can predict
            obs, _ = env.reset()
            action, _ = model.predict(obs)
            print(f"   🎯 Model prediction works, action shape: {action.shape}")
            
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            raise
    
    # Test model creation with CRL disabled
    print(f"\n📋 Testing model creation with CRL disabled:")
    
    crl_config = config.get_crl_config("balanced")
    crl_config["use_crl"] = False  # Disable CRL
    
    try:
        model = CustomTD3_AC(
            SimpleMlpTD3Policy,
            env,
            verbose=0,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            policy_kwargs=policy_kwargs,
            batch_size=64,
            buffer_size=1000,
            learning_starts=10,
            use_crl=False,  # Explicitly disable
            action_reg_coef=0.1,  # Fixed penalty
        )
        
        crl_stats = model.get_crl_stats()
        assert not crl_stats['crl_enabled'], "CRL should be disabled"
        print(f"   ✅ CRL disabled model works correctly")
        
    except Exception as e:
        print(f"   ❌ CRL disabled model creation failed: {e}")
        raise
    
    env.close()
    print("✅ CRL model creation tests passed!")


def test_crl_lambda_updates():
    """Test that lambda updates work correctly."""
    print("\n🧪 Testing CRL lambda update mechanism...")
    
    # Create a simple model for testing
    env = PortfolioEnv(reward_type="TRANSACTION_COST")
    
    policy_kwargs = {
        "net_arch": {"pi": [64, 32], "qf": [64, 32]},  # Smaller networks for testing
        "actor_lr": 1e-4,
        "critic_lr": 1e-3
    }
    
    model = CustomTD3_AC(
        SimpleMlpTD3Policy,
        env,
        verbose=0,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        policy_kwargs=policy_kwargs,
        batch_size=32,
        buffer_size=100,
        learning_starts=5,
        use_crl=True,
        constraint_threshold=0.05,
        lambda_lr=1e-3,
        initial_lambda=0.1,
    )
    
    # Test lambda update with constraint violation
    initial_lambda = model.lambda_param.item()
    print(f"   📊 Initial λ: {initial_lambda:.4f}")
    
    # Simulate constraint violation (cost > threshold)
    high_cost = 0.1  # Higher than threshold of 0.05
    model._update_lambda(high_cost)
    after_violation_lambda = model.lambda_param.item()
    print(f"   📊 λ after violation (cost={high_cost:.3f}): {after_violation_lambda:.4f}")
    
    # Lambda should increase when constraint is violated
    assert after_violation_lambda > initial_lambda, f"Lambda should increase after violation: {after_violation_lambda} <= {initial_lambda}"
    print(f"   ✅ Lambda correctly increased after constraint violation")
    
    # Simulate constraint satisfaction (cost < threshold)
    low_cost = 0.01  # Lower than threshold of 0.05
    model._update_lambda(low_cost)
    after_satisfaction_lambda = model.lambda_param.item()
    print(f"   📊 λ after satisfaction (cost={low_cost:.3f}): {after_satisfaction_lambda:.4f}")
    
    # Lambda should decrease when constraint is satisfied
    assert after_satisfaction_lambda < after_violation_lambda, f"Lambda should decrease after satisfaction: {after_satisfaction_lambda} >= {after_violation_lambda}"
    print(f"   ✅ Lambda correctly decreased after constraint satisfaction")
    
    # Test that lambda stays non-negative
    assert model.lambda_param.item() >= 0, f"Lambda should never be negative: {model.lambda_param.item()}"
    print(f"   ✅ Lambda remains non-negative")
    
    env.close()
    print("✅ CRL lambda update tests passed!")


def main():
    """Run all CRL integration tests."""
    print("🚀 Starting CRL Integration Tests...")
    print("=" * 60)
    
    try:
        # Test 1: Configuration loading
        test_crl_config()
        
        # Test 2: Model creation
        test_crl_model_creation()
        
        # Test 3: Lambda update mechanism
        test_crl_lambda_updates()
        
        print("\n" + "=" * 60)
        print("🎉 ALL CRL INTEGRATION TESTS PASSED!")
        print("\n✨ CRL integration is working correctly!")
        print("\n🎯 You can now use CRL with commands like:")
        print("   python train_simple_mlp.py --algorithm TD3 --crl-profile balanced")
        print("   python train_simple_mlp.py --algorithm TD3 --crl-profile conservative")
        print("   python train_simple_mlp.py --algorithm TD3 --crl-profile aggressive")
        print("   python train_simple_mlp.py --algorithm TD3 --disable-crl")
        
    except Exception as e:
        print(f"\n❌ CRL INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 