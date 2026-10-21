import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Training configuration
N_EPISODES = 10000

class QLearningAgent:
    def __init__(self, learning_rate=0.9,
                 discount_factor=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Each state has two Q-values:
        # index 0 = Stick, index 1 = Hit
        self.q_table = defaultdict(lambda: np.zeros(2))

    def get_action(self, state, training=True):
        # Explore: choose a random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)

        # Exploit: choose the action with the highest Q-value
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward,
            next_state, terminated):

        current_q = self.q_table[state][action]

        if terminated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        self.q_table[state][action] += (
            self.lr * (target_q - current_q)
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

def train_agent(n_episodes=N_EPISODES):
    env = gym.make("Blackjack-v1")
    agent = QLearningAgent()

    episode_rewards = []
    progress_interval = max(1, n_episodes // 10)

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )
            done = terminated or truncated

            agent.update(
                state, action, reward, next_state, done
            )

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % progress_interval == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode + 1}/{n_episodes}, "
                f"Average reward (last 100): {recent_avg:.3f}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    return agent, episode_rewards

def plot_results(episode_rewards):
    cumulative_average = (
        np.cumsum(episode_rewards)
        / np.arange(1, len(episode_rewards) + 1)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        range(1, len(episode_rewards) + 1),
        cumulative_average,
        label="Cumulative average reward"
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.title("Q-Learning on Gymnasium Blackjack")
    plt.xlim(1, len(episode_rewards))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("blackjack_training.png", dpi=150)
    plt.show()

def test_agent(agent, n_episodes=1000):
    env = gym.make("Blackjack-v1")

    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Use the learned policy without exploration
            action = agent.get_action(state, training=False)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )
            done = terminated or truncated

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    rewards = np.array(rewards)

    print("\n--- Final Agent Evaluation ---")
    print(f"Test episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Win rate: {np.mean(rewards == 1):.2%}")
    print(f"Draw rate: {np.mean(rewards == 0):.2%}")
    print(f"Loss rate: {np.mean(rewards == -1):.2%}")



if __name__ == "__main__":
    agent, episode_rewards = train_agent()
    plot_results(episode_rewards)
    test_agent(agent, n_episodes=1000)
