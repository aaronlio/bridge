import numpy as np
def SFD_train(length):
    wheel_positions = [0, 176, 340, 516, 680, 856]
    support_a_position = 15
    support_b_position = 1090
    reaction_forces = [[],[]]
    for i in range(length-wheel_positions[5]):
        reaction_b = 0
        reaction_a = 0
        relative_wheel_positions = list(map(lambda x: x - support_a_position, wheel_positions))
        reaction_b = (sum(relative_wheel_positions)*(400/6))/(support_b_position-support_a_position)
        reaction_a = 400-reaction_b
        
        wheel_positions = list(map(lambda x: x + 1, wheel_positions))
        reaction_forces[0].append(reaction_a)
        reaction_forces[1].append(reaction_b)
        print(reaction_forces)

    print(max(reaction_forces[0]), max(reaction_forces[1]))
SFD_train(1280)


