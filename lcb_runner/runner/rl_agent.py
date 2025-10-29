import torch
import torch.optim as optim
from torch.distributions import Categorical

from lcb_runner.runner.policy_network import PolicyNetwork, get_state_embedding, STATE_EMBEDDING_DIM

class RLAgent:
    """
    The REINFORCE agent. Now includes device management for GPU support.
    """
    def __init__(self, action_dim, learning_rate=0.01, gamma=0.99):
        # --- 1. GPU Support ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RLAgent is using device: {self.device}")

        self.gamma = gamma
        
        # The policy network is now initialized with the correct input dimension and device
        self.policy = PolicyNetwork(input_dim=STATE_EMBEDDING_DIM, action_dim=action_dim, device=self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.rewards = []
        self.log_probs = []

    def select_action(self, state: dict):
        """
        Selects an action based on the current policy and state.
        The state embedding is now created on the correct device.
        """
        state_embedding = get_state_embedding(state, self.device)
        
        action_probs = self.policy(state_embedding)
        
        m = Categorical(action_probs)
        action = m.sample()
        
        self.log_probs.append(m.log_prob(action))
        
        return action.item()

    def store_reward(self, reward: float):
        """Stores the reward for the last action."""
        self.rewards.append(reward)

    def update_policy(self):
        """
        Updates the policy network using REINFORCE.
        This is called at the end of an episode (all repair attempts for one problem).
        """
        if not self.log_probs:
            return

        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.tensor(discounted_rewards, device=self.device)
        # Normalize for stability
        std = discounted_rewards.std()
        if std > 1e-6:  # Only normalize if there's variation
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / std
        # else: keep as is (all rewards are the same)

        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.clear_memory()

    def clear_memory(self):
        """Resets memory for the next episode."""
        del self.rewards[:]
        del self.log_probs[:]