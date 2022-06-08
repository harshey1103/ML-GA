import numpy as np
import pandas as pd
import random
import h2o
h2o.init()

#parameters that can be tuned :
"""
fitness multiplier 
range of random children genereator
number of children 
margin 
"""

## pandas->int->h2o data frame start
fields = ['Eop viscosity', 'Hypo viscosity', 'Hypo-1 Solution flow', 'Viscosity']
pd_df = pd.read_csv('flat_file_new.csv', usecols=fields)
pd_df = pd_df.astype('int')
df = h2o.H2OFrame(pd_df)
## pandas data frame start


## H2O ML model start
from h2o.estimators import H2OGradientBoostingEstimator

df["Viscosity"] = df["Viscosity"].asfactor()
predictors = ["Eop viscosity", "Hypo viscosity", "Hypo-1 Solution flow"]
response = "Viscosity"
train, valid = df.split_frame(ratios=[.8])

# Build and train the model:
pros_gbm = H2OGradientBoostingEstimator(nfolds=5, seed=1111, keep_cross_validation_predictions = True)
pros_gbm.train(x=predictors, y=response, training_frame=train)

# Eval performance:
perf = pros_gbm.model_performance()

# Generate predictions on a test set (if necessary):
pred_valid = pros_gbm.predict(valid)
## H2O ML model end


## ML output function start
def ML_model(output):
    data = {'Eop viscosity': [589], 'Hypo viscosity': [533], 'Hypo-1 Solution flow': [output]}  
    inp_df = pd.DataFrame(data)  


    inp_df = inp_df.astype('int')   
    inp_df = h2o.H2OFrame(inp_df)
    pred = pros_gbm.predict(inp_df)
    pred = int(pred[0, 0])
    return pred 
## ML output function start


# stats of viscosity 
std_dev_flow = pd_df["Hypo-1 Solution flow"].std()
# bound for output
max_val_flow = pd_df["Hypo-1 Solution flow"].max() + std_dev_flow/2
min_val_flow = pd_df["Hypo-1 Solution flow"].min() - std_dev_flow/2
min_val_flow = max(0, min_val_flow)


## fitness func start
desired_viscosity = 450
def fitness_func(output):
    ####
    pred = ML_model(output)
    ####
    fitness = 50.0 / (np.abs(desired_viscosity - pred)+0.000000001)
    return fitness
## fitness func start


## generating children start
def offsprings(par):
    child = random.uniform(max(0, par-std_dev_flow/2), par+std_dev_flow/2)	
    return child
## generating children end


def main():

    found = False
    num_children = 20

    #initiate population & generation
    population = []
    population.append(random.uniform(min_val_flow, max_val_flow))
    generation = 1

    # margin for fitness
    margin = 10

    while not found:
        for _ in range (num_children) : 
            population.append(offsprings(population[-1]))

        population = sorted(population, key = fitness_func)

        # termniation after fitness achieved
        if fitness_func(population[-1]) >= margin : 
            found = True
            break

        #### termination of loop 
        if generation == 5:
            break
        ####
        generation = generation+1

    print('output = ' + str(population[-1]))
    print('fintess value of output = ' + str(fitness_func(population[-1])))
    print('number of generations = ' + str(generation))

main()
h2o.cluster().shutdown()


# AWS lambda
# 3 train validation random
# run 15 times (15 different values of viscosities)
# append in new csv file 
