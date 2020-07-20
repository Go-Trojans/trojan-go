from algos.utils import set_gpu_memory_target, load_model_from_disk

import numpy as np

        
def main():
    gpu_frac = 0.20
    set_gpu_memory_target(gpu_frac)
    agent_model = ("./checkpoints/iteration_Savedmodel/initial.json", "./checkpoints/iteration_Savedmodel/initial.h5")
    model = load_model_from_disk(agent_model)
    model.summary()

    model_input = []

    for _ in range(100):
        board_tensor = np.random.randint(0, 3, size=(7, 5, 5))
        model_input.append(board_tensor)

    model_input = np.array(model_input) 
     

    action_target = []
    for _ in range (100):
        search_prob = np.random.randn(26)
        #search_prob_flat = search_prob.reshape(25,)
        action_target.append(search_prob)
        
    action_target = np.array(action_target)    


    value_target = np.random.rand(100)
    value_target = np.array(value_target) 

    

    X = model_input[0]
    X = np.expand_dims(X, axis=0)
    prediction = model.predict(X)
    print(prediction)
    



if __name__ == '__main__':
    main()
