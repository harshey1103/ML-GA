import numpy as np
import pandas as pd
import random
import sys
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
pros_gbm = H2OGradientBoostingEstimator(nfolds=3, seed=13, keep_cross_validation_predictions = True)
pros_gbm.train(x=predictors, y=response, training_frame=train)

# Eval performance:
perf = pros_gbm.model_performance()

# Generate predictions on a test set (if necessary):
pred_valid = pros_gbm.predict(valid)
## H2O ML model end


## ML output function start
var1 = 0
var2 = 0
def ML_model(output):
    data = {'Eop viscosity': [var1], 'Hypo viscosity': [var2], 'Hypo-1 Solution flow': [output]}  
    inp_df = pd.DataFrame(data)  


    inp_df = inp_df.astype('int')   
    inp_df = h2o.H2OFrame(inp_df)
    pred = pros_gbm.predict(inp_df)
    pred = int(pred[0, 0])
    return pred 
## ML output function start

## Statistics start
# stats of flow
std_dev_flow = pd_df["Hypo-1 Solution flow"].std()
# bound for output
max_val_flow = pd_df["Hypo-1 Solution flow"].max()  - std_dev_flow/2
min_val_flow = pd_df["Hypo-1 Solution flow"].min()  + std_dev_flow/2
min_val_flow = max(0, min_val_flow)
# stats of Eop viscosity
max_val_eopViscosity = pd_df["Eop viscosity"].max()
min_val_eopViscosity = pd_df["Eop viscosity"].min()
# stats of Hypo viscosity
max_val_hypoViscosity = pd_df["Hypo viscosity"].max()
min_val_hypoViscosity = pd_df["Hypo viscosity"].min()
## Statistics end


## fitness func start
desired_viscosity = 450
def fitness_func(output):
    ####
    pred = ML_model(output)
    ####
    fitness = 100.0 / (np.abs(desired_viscosity - pred)+0.0001)
    return fitness
## fitness func start


## generating children start
def offsprings(par):
    left = max(min_val_flow, par-std_dev_flow/2)
    right = min(max_val_flow, par+std_dev_flow/2)
    child = random.uniform(left, right)	
    return child
## generating children end

#

ouput_df = pd.DataFrame({'Eop viscosity': [], 'Hypo viscosity': [], 'Hypo-1 Solution flow': [], 'Viscosity': [], 'fitness': [], 'generation': []})
def main():

    found = False
    num_children = 10

    #initiate population & generation
    population = []
    population.append(random.uniform(min_val_flow, max_val_flow))

    # assign var1 and var2
    var1 = random.uniform(min_val_eopViscosity, max_val_eopViscosity)
    var2 = random.uniform(min_val_hypoViscosity, max_val_hypoViscosity)

    #declare generation
    generation = 1
    # margin for fitness
    margin = 5

    while not found:
        for _ in range (num_children) : 
            population.append(offsprings(population[-1]))

        population = sorted(population, key = fitness_func)

        # termniation after fitness achieved
        if fitness_func(population[-1]) >= margin : 
            found = True
            break

        #### termination of loop 
        if generation == 7:
            found = True
            break
        ####
        generation = generation+1
    
    # print or store output
    global ouput_df
    temp_df = pd.DataFrame({'Eop viscosity': [var1], 'Hypo viscosity': [var2], 'Hypo-1 Solution flow': [population[-1]], 'Viscosity': [ML_model(population[-1])], 'fitness': [fitness_func(population[-1])], 'generation': [generation]})
    ouput_df = pd.concat([ouput_df, temp_df])

for i in range (5):
    main()

ouput_df.to_excel("output3.xlsx")    

h2o.cluster().shutdown()

# output wider range of output; num child = 10
# ouput 1 output range to min max; num child = 20
# output 2 output range narrower; num child = 20