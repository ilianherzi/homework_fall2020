# for implementing random selection of top k        

class TopkPolicy(object):

    def __init__(self, critic, k = 5):
        self.critic = critic
        # could also use distribution around the highest prob action. For now it's just a uniform random variable

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        _, actions = self.critic.qa_values(observation).topk()
        actions = actions.squeeze()
        random_indices = torch.from_numpy(
            np.random.randint(0, actions.size(1), actions.size(0))
        ).unsqueeze(1)
        action = torch.gather(actions,
                     1, 
                    random_indices).squeeze(1)
        return action

