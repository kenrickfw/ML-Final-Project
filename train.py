from main import RainbowAgent
from snakegame import SnakeGameAI
from plotgraph import plot
import torch

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    agent = RainbowAgent(input_size=11, hidden_size=256, output_size=3, n_steps=3)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.remember(state_old, final_move, reward, state_new, done)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                torch.save(agent.model.state_dict(), "model.pth")

            if agent.n_games % 10 == 0:
                agent.update_target_model()

            print(f"Game {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
