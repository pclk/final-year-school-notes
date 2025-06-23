r"""°°°
# Reinforcement Learning Example 2


°°°"""
# |%%--%%| <G8CyeUKw8N|vhqrKIQHbP>
r"""°°°
Set up the Reward table 

reward -1 where states are not possible

reward 0 where state are possible but not goal

reward -50 where states are at the obstacle

reward 100 where state is the goal

![maze.png](data:image/png;base64,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)

°°°"""
# |%%--%%| <vhqrKIQHbP|cGp5TGAGdU>

import numpy as np

# Reward Table matrix
#Add code


# |%%--%%| <cGp5TGAGdU|Dg40VL6Q3C>
r"""°°°
Set the row and column in the Q table to intialise.
°°°"""
# |%%--%%| <Dg40VL6Q3C|OG5X4NPkgO>

# Q Table matrix
Q = np.matrix(np.zeros([#Add code]))

# |%%--%%| <OG5X4NPkgO|jFLJ7HAlci>

# Gamma (learning parameter).
gamma = 0.8


# |%%--%%| <jFLJ7HAlci|60vWqMHlOk>
r"""°°°
Set the intial state at rando value
°°°"""
# |%%--%%| <60vWqMHlOk|ZFSTZbrXHw>

# Initial state. (Usually to be chosen at random)
initial_state = #Add code


# |%%--%%| <ZFSTZbrXHw|CtmPsfer7o>

# Initial state. (Usually to be chosen at random)
initial_state = 5

# |%%--%%| <CtmPsfer7o|VjNCXSWcX5>

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    # print(av_act)
    return av_act

# Get available actions in the current state
available_act = available_actions(initial_state) 


# |%%--%%| <VjNCXSWcX5|35dJjCgA7y>

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    # print(next_action)
    return next_action

# Sample next action to be performed
action = sample_next_action(available_act)


# |%%--%%| <35dJjCgA7y|dw2OxR4Uv6>

# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm
def update(current_state, action, gamma):
    
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    # print(max_index)

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma * max_value

# Update Q matrix
update(initial_state,action,gamma)

# |%%--%%| <dw2OxR4Uv6|xs10YYUxWE>

# Training

print("Before training Q matrix:")
print(Q)

# |%%--%%| <xs10YYUxWE|WpVjO2d2E5>
r"""°°°
Set the iterations for training
°°°"""
# |%%--%%| <WpVjO2d2E5|d0EXk8rFMi>

# Train over 10 000 iterations. (Re-iterate the process above).
for i in range(#Add code):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,gamma)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

# |%%--%%| <d0EXk8rFMi|zd5pzq0Pdj>
r"""°°°
Set the Goal state

Set the current state
°°°"""
# |%%--%%| <zd5pzq0Pdj|66slIQcaR7>

# Testing-Agent interact with environment

# Goal state = 8

current_state = #Add code
steps = [current_state]


while current_state != #Add code:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)


# |%%--%%| <66slIQcaR7|ffUpTalUj0>

# Testing-Agent interact with environment

# Goal state = 8

current_state = 11
steps = [current_state]


while current_state != 8:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)

