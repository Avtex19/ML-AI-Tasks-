import math
import random
import time


class Nim:

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Create a new game with an initial configuration of piles.
        """
        self.piles = initial[:]
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Return a set of all valid (pile, count) moves for a given state.
        """
        return {(i, j) for i, pile in enumerate(piles) for j in range(1, pile + 1)}

    @classmethod
    def other_player(cls, player):
        """
        Given a current player, return the other one.
        """
        return 1 - player

    def switch_player(self):
        """
        Toggle the active player.
        """
        self.player = self.other_player(self.player)

    def move(self, action):
        """
        Apply an action to the game board.
        """
        i, remove = action

        if self.winner is not None:
            raise ValueError("Game has already concluded")
        if not (0 <= i < len(self.piles)) or not (1 <= remove <= self.piles[i]):
            raise ValueError("Invalid move attempted")

        self.piles[i] -= remove
        self.switch_player()

        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI:

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize the AI agent with Q-learning memory and parameters.
        """
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Learn by updating Q-values for a transition and reward.
        """
        prior = self.get_q_value(old_state, action)
        best = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, prior, reward, best)

    def get_q_value(self, state, action):
        """
        Return Q-value for a (state, action) pair, defaulting to 0.
        """
        key = (tuple(state), tuple(action))
        return self.q.get(key, 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Apply Q-learning formula to update Q-value estimate.
        """
        estimate = reward + future_rewards
        updated = old_q + self.alpha * (estimate - old_q)
        self.q[(tuple(state), tuple(action))] = updated
        return updated

    def best_future_reward(self, state):
        """
        Return the highest possible Q-value from any legal action.
        """
        options = Nim.available_actions(state)
        if not options:
            return 0

        values = [self.get_q_value(state, action) for action in options]
        return max(values)

    def choose_action(self, state, epsilon=True):
        """
        Select an action using epsilon-greedy strategy.
        """
        actions = list(Nim.available_actions(state))
        if not actions:
            return None

        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        best_value = float('-inf')
        best_actions = []

        for action in actions:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return random.choice(best_actions)


def train(n):
    """
    Play `n` self-games to train AI using reinforcement learning.
    """
    ai = NimAI()
    for game_num in range(n):
        print(f"Training game {game_num + 1}")
        game = Nim()
        memory = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            current_state = game.piles[:]
            move = ai.choose_action(current_state)

            memory[game.player]["state"] = current_state
            memory[game.player]["action"] = move

            game.move(move)
            new_state = game.piles[:]

            if game.winner is not None:
                ai.update(current_state, move, new_state, -1)
                ai.update(
                    memory[game.player]["state"],
                    memory[game.player]["action"],
                    new_state,
                    1
                )
                break
            elif memory[game.player]["state"] is not None:
                ai.update(
                    memory[game.player]["state"],
                    memory[game.player]["action"],
                    new_state,
                    0
                )
    print("Training complete.")
    return ai


def play(ai, human_player=None):
    """
    Let a human play against a trained AI.
    """
    if human_player not in (0, 1):
        human_player = random.randint(0, 1)

    game = Nim()

    while True:
        print("\nCurrent Piles:")
        for idx, count in enumerate(game.piles):
            print(f"Pile {idx}: {count}")

        time.sleep(1)
        legal_moves = Nim.available_actions(game.piles)

        if game.player == human_player:
            print("Your Move")
            while True:
                try:
                    pile = int(input("Choose pile: "))
                    count = int(input("Choose count: "))
                    if (pile, count) in legal_moves:
                        break
                    else:
                        print("Invalid move, try again.")
                except Exception:
                    print("Invalid input. Try again.")
        else:
            print("AI's Move")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI removes {count} from pile {pile}")

        game.move((pile, count))

        if game.winner is not None:
            print("\nGAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"{winner} wins!")
            return
