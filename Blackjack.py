import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearningAgent:
    def __init__(self, learning_rate=0.1,
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


def play_blackjack():
    # Initialize the Blackjack environment
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    
    # Reset the game to get the initial state
    # state is a tuple: (player_sum, dealer_showing, usable_ace)
    state, info = env.reset()
    done = False

    print("\nBlackjack")

    while not done:
        player_sum, dealer_card, usable_ace = state
        print(f"\nYour current hand total: {player_sum} (Usable Ace: {usable_ace})")
        print(f"Dealer is showing: {dealer_card}")
        
        # Get human input from terminal
        action_input = input("Choose action - [H]it or [S]tick: ").strip().lower()
        
        if action_input == 'h':
            action = 1  # 1 maps to Hit in Gymnasium
        elif action_input == 's':
            action = 0  # 0 maps to Stick/Stand in Gymnasium
        else:
            print("Invalid input. Please type 'H' or 'S'.")
            continue
            
        # Step the environment forward based on choice
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Game Over Summary
    player_sum, dealer_card, usable_ace = state
    print("\n=== Game Over ===")
    print(f"Your final score: {player_sum}")
    
    if reward > 0:
        print("Win")
    elif reward < 0:
        print("Lose")
    else:
        print("Draw")
        
    env.close()

if __name__ == "__main__":
    play_blackjack()
