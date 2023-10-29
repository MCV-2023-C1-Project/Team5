import optuna
import random

def objective(trial):
    weight_names = ["w1", "w2", "w3"]  # define the names of the weights
    weights = {}
    remainder = 1  # total energy of the weighted sum coefficients

    for _ in range(len(weight_names)-1):
        weight_name = random.choice(weight_names)  # start from random weight
        weight_names.remove(weight_name)  # don't use it in following iters

        # use remainder as the suggest upper boundd to keep total energy <=1
        value = trial.suggest_float(weight_name, 0, remainder, step=0.1)
        weights[weight_name] = value
        remainder -= value  # update remainder
    # last weight should be suggested outside the loop,
    # since it must be equal to the remainder to complete all weights
    # to sum up to 1
    value = trial.suggest_float(weight_names[0], remainder, remainder)
    weights[weight_names[0]] = value    
    
    # just a random formula to return something during study
    # here must be our retrieval code instead, namely the combination
    # of losses part
    result = weights["w1"] + weights["w2"] ** 2 + weights["w3"] ** 3
    return result


study = optuna.create_study(storage="sqlite:///test1.db", study_name="v1")
study.optimize(objective, n_trials=25)

