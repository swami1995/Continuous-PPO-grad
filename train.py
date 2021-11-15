import torch
import numpy as np
import time
from running_mean_std import RunningMeanStd
from test import evaluate_model
from tensorboardX import SummaryWriter
import ipdb

class Train:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.gamma = 0.99

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))

        self.running_reward = 0

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs, value_grads):
        states = states.view(-1, states.shape[-1]).detach()
        actions = actions.view(-1, actions.shape[-1]).detach()
        returns = returns.view(-1, returns.shape[-1]).detach()
        advs = advs.view(-1, advs.shape[-1]).detach()
        values = values.view(-1, values.shape[-1]).detach()
        value_grads = value_grads.view(-1, value_grads.shape[-1]).detach()
        log_probs = log_probs.view(-1, log_probs.shape[-1]).detach()
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices],\
                  log_probs[indices], value_grads[indices]

    def train(self, states, actions, advs, values, log_probs, value_grads):

        values = torch.stack(values[:-1])
        value_grads = torch.stack(value_grads)
        log_probs = torch.stack(log_probs)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        actions = torch.stack(actions)
        for epoch in range(self.epochs):
            for state, action, return_, adv, old_value, old_log_prob, value_grad_targ in self.choose_mini_batch(self.mini_batch_size,
                                                                                               states, actions, returns,
                                                                                               advs, values, log_probs, 
                                                                                               value_grads):
                # state = torch.Tensor(state).to(self.agent.device)
                # action = torch.Tensor(action).to(self.agent.device)
                # return_ = torch.Tensor(return_).to(self.agent.device)
                # adv = torch.Tensor(adv).to(self.agent.device)
                # old_value = torch.Tensor(old_value).to(self.agent.device)
                # old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)

                state = state.to(self.agent.device).requires_grad_(True)
                action = action.to(self.agent.device)
                # if not self.agent.continuous:
                #     action = action.squeeze()
                return_ = return_.to(self.agent.device)
                adv = adv.to(self.agent.device)
                old_value = old_value.to(self.agent.device)
                old_log_prob = old_log_prob.to(self.agent.device)
                value_grad_targ = value_grad_targ.to(self.agent.device)


                value = self.agent.critic(state)
                value_grad = torch.autograd.grad(value.sum(), state, retain_graph=True, create_graph=True)[0]
                # clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                # clipped_v_loss = (clipped_value - return_).pow(2)
                # unclipped_v_loss = (value - return_).pow(2)
                # critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                # ipdb.set_trace()

                critic_loss = self.agent.critic_losses(value, return_, value_grad, value_grad_targ)

                new_log_prob = self.calculate_log_probs(self.agent.current_policy, state, action)

                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss

    def step(self):
        state = self.env.reset()
        for iteration in range(1, 1 + self.n_iterations):
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            value_grads = []
            state = self.env.reset()
            self.start_time = time.time()
            next_value = self.agent.critic(state)
            for t in range(self.horizon):
                # self.state_rms.update(state)
                # state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                # dist = self.agent.choose_dist(state)
                dist = self.agent.current_policy(state)
                action = dist.sample()#.cpu().numpy()[0]
                # action = np.clip(action, self.agent.action_bounds[0], self.agent.action_bounds[1])
                log_prob = dist.log_prob(action)
                # value = self.agent.get_value(state)
                value = next_value.clone()
                state.requires_grad_(True)
                next_state, reward, _, _ = self.env.step(action, state=state, return_costs=True)
                next_value = self.agent.critic(next_state)
                value_grad = torch.autograd.grad(next_value.sum(),state, retain_graph=True)[0]
                reward_grad = torch.autograd.grad(reward.sum(),state, retain_graph=True)[0]
                done = False
                # print(reward.mean())
                states.append(state)
                actions.append(action)
                rewards.append(reward.unsqueeze(-1))
                values.append(value)
                log_probs.append(log_prob)
                value_grads.append(reward_grad + value_grad*self.gamma)
                dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            # next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            # next_value = self.agent.get_value(next_state) * (1 - done)
            # next_value = self.agent.critic(next_state) * (1-done)
            values.append(next_value)
            # ipdb.set_trace()
            advs = self.get_gae(rewards, values, dones, self.gamma)
            states = torch.stack(states)
            actor_loss, critic_loss = self.train(states, actions, advs, values, log_probs, value_grads)
            # self.agent.set_weights()
            self.agent.schedule_lr()
            # eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms, self.agent.action_bounds)
            # self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, torch.stack(rewards).sum(dim=0).mean())#, eval_rewards)
        self.eval_and_save()

    def eval_and_save(self,):
        self.env.batch_size = 512
        state = self.env.reset()
        for iteration in range(1, 120):
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []

            state = self.env.reset()#.cpu()
            # ipdb.set_trace()
            for t in range(self.horizon):
                # self.state_rms.update(state)
                # state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                # dist = self.agent.choose_dist(state)
                dist = self.agent.current_policy(state)
                action = dist.sample()#.cpu().numpy()[0]
                # action = np.clip(action, self.agent.action_bounds[0], self.agent.action_bounds[1])
                log_prob = dist.log_prob(action)#torch.Tensor(action))
                # value = self.agent.get_value(state)
                value = self.agent.critic(state)
                # next_state, reward, done, _ = self.env(state, action)
                next_state, reward, _ = self.env(state, action, return_costs=True)
                done = False
                if not self.agent.continuous:
                    action = action.unsqueeze(-1)

                states.append(state)
                actions.append(action)
                rewards.append(reward.unsqueeze(-1))
                values.append(value)
                # log_probs.append(log_prob)
                # dones.append(done)

                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            print("evaluation return: ", rewards_arr.sum(dim=1).mean())
            # done = dones[-1] = True
            states_arr = torch.stack(states)
            actions_arr = torch.stack(actions)
            rewards_arr = torch.stack(rewards)
            values_arr = torch.stack(values)
            if iteration==1:
                states_buffer = states_arr
                actions_buffer = actions_arr
                rewards_buffer = rewards_arr
                values_buffer = values_arr
            else:
                states_buffer = torch.cat([states_buffer, states_arr], dim=1)
                actions_buffer = torch.cat([actions_buffer, actions_arr], dim=1)
                rewards_buffer = torch.cat([rewards_buffer, rewards_arr], dim=1)
                values_buffer = torch.cat([values_buffer, values_arr], dim=1)

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0

        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return torch.stack(advs)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards):
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 2 == 0:
            # ipdb.set_trace()
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:.3f}| "
                  f"Running_reward:{self.running_reward:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            # self.agent.save_weights(iteration, self.state_rms)

        # with SummaryWriter(self.env_name + "/logs") as writer:
        #     writer.add_scalar("Episode running reward", self.running_reward, iteration)
        #     writer.add_scalar("Episode reward", eval_rewards, iteration)
        #     writer.add_scalar("Actor loss", actor_loss, iteration)
        #     writer.add_scalar("Critic loss", critic_loss, iteration)
