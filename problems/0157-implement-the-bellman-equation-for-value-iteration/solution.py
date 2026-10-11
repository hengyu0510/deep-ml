import torch

def bellman_update(V: torch.Tensor, transitions: list, gamma: float) -> torch.Tensor:
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: torch.Tensor, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      torch.Tensor, updated state values
    """
    # TODO: Implement Bellman update

    
    # 注意到如果done是1，那么后面就没有V(s')了，说明任务已经结束了
    num_states = V.shape[0]
    V_next =torch.zeros(num_states)
    for i in range(num_states): 
      # Ways to iterate through a dictionary's key-value 
      value_a = []
      for a , outcomes in transitions[i].items():
        values = 0
        for prob, next_state, reward, done in outcomes:
          values += prob * (reward + gamma * V[next_state]*(1-done))
        value_a.append(values)
      V_next[i]=max(value_a)
    return V_next