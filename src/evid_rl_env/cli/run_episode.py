from evid_rl_env.environment.environment import ClaimEnv
from evid_rl_env.data.dataset import load_dataset
from evid_rl_env.agent.policy import SoftmaxPolicy, ActorCriticPolicy
from evid_rl_env.environment.actions import ACTIONS

import argparse


def run(policy='actor'):
    dataset = load_dataset()

    env = ClaimEnv(dataset)

    if policy == 'actor':
        policy = ActorCriticPolicy(len(list(ACTIONS)))
    elif policy == 'softmax':
        policy = SoftmaxPolicy(len(list(ACTIONS)))

    state = env.reset()
    done = False

    step = 0

    print("\n=== START EPISODE ===")
    print("CLAIM:", state.claim)

    while not done:
        action, payload = policy.act(state)
        state, reward, done, _ = env.step(action, payload)

        step += 1

        print(f"\n--- STEP {step} ---")
        print("Action:", action)
        print("Payload:", payload)
        print("Selected:", [e.id for e in state.selected_evidence])
        print("Debate:", state.debate_history)

    print("\n=== FINAL ===")
    print("Steps:", step)
    print("Final Reward:", round(reward, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default='actor')
    args = parser.parse_args()

    run(args.policy)

if __name__ == "__main__":
    main()