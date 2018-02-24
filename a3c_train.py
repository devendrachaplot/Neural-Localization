import torch.optim as optim
import logging

from maze2d import *
from model import *
from collections import deque


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(
            model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model):
    args.seed = args.seed + rank
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = Maze2D(args)
    action_hist_size = args.hist_size

    model = Localization_2D_A3C(args)
    if (args.load != "0"):
        print("Training thread: {}, Loading model {}".format(rank, args.load))
        model.load_state_dict(torch.load(args.load))
    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)
    model.train()

    values = []
    log_probs = []
    p_losses = []
    v_losses = []

    episode_length = 0
    num_iters = 0
    done = True

    state, depth = env.reset()
    state = torch.from_numpy(state)
    while num_iters < args.num_iters/1000:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            # filling action history with action 3 at the start of the episode
            action_hist = deque(
                [3] * action_hist_size,
                maxlen=action_hist_size)
            episode_length = 0
        else:
            action_hist.append(action)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            state = state.float()
            ax = Variable(torch.from_numpy(np.array(action_hist)))
            dx = Variable(torch.from_numpy(np.array([depth])).long())
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())

            value, logit = model(
                (Variable(state.unsqueeze(0)), (ax, dx, tx)))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data

            log_prob = log_prob.gather(1, Variable(action))

            action = action.numpy()[0, 0]

            state, reward, done, depth = env.step(action)
            done = done or episode_length >= args.max_episode_length

            if done:
                episode_length = 0
                state, depth = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        state = state.float()
        if not done:
            action_hist.append(action)
            ax = Variable(torch.from_numpy(np.array(action_hist)))
            dx = Variable(torch.from_numpy(np.array([depth])).long())
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())
            value, _ = model((Variable(state.unsqueeze(0)), (ax, dx, tx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])

        if(len(p_losses) > 1000):
            num_iters += 1
            print(" ".join([
                  "Training thread: {:2d},".format(rank),
                  "Num iters: {:4d}K,".format(num_iters),
                  "Avg policy loss: {0:+.3f},".format(np.mean(p_losses)),
                  "Avg value loss: {0:+.3f}".format(np.mean(v_losses))]))
            logging.info(" ".join([
                  "Training thread: {:2d},".format(rank),
                  "Num iters: {:4d}K,".format(num_iters),
                  "Avg policy loss: {0:+.3f},".format(np.mean(p_losses)),
                  "Avg value loss: {0:+.3f}".format(np.mean(v_losses))]))
            p_losses = []
            v_losses = []

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

    print("Training thread {} completed".format(rank))
    logging.info("Training thread {} completed".format(rank))
