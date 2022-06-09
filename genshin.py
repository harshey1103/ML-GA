def genshin(pd_df):
    import pandas as pd
    import numpy as np
    import random
    import h2o
    h2o.init()

    #parameters that can be tuned :
    """
    fitness multiplier (currently set to 100)
    margin 
    range of random children genereator
    number of children 
    """


    ## pandas->int->h2o data frame start
    fields = ['Eop viscosity', 'Hypo viscosity', 'Hypo-1 Solution flow', 'Viscosity']
    pd_df = pd_df.filter(fields)
    pd_df = pd.DataFrame(pd_df)
    pd_df = pd_df.astype('int')
    df = h2o.H2OFrame(pd_df)
    ## pandas data frame end




    ## H2O ML model start
    from h2o.estimators import H2OGradientBoostingEstimator

    df["Viscosity"] = df["Viscosity"].asfactor()
    predictors = ["Eop viscosity", "Hypo viscosity", "Hypo-1 Solution flow"]
    response = "Viscosity"
    train, valid = df.split_frame(ratios=[.8])

    # Build and train the model:
    pros_gbm = H2OGradientBoostingEstimator(nfolds=3, seed=13, keep_cross_validation_predictions = True)
    pros_gbm.train(x=predictors, y=response, training_frame=train)

    ## Eval performance:
    # perf = pros_gbm.model_performance()

    ## Generate predictions on a test set (if necessary):
    # pred_valid = pros_gbm.predict(valid)
    ## H2O ML model end

    global dataframe1
    dataframe1 = pd.DataFrame({'Eop viscosity': [], 'Hypo viscosity': [], 'Actual Hypo-1 Solution flow': [], 'Actual Viscosity': [], 'Predicted Hypo -1 solution  flow': [], 'Predicted Viscosity': []})
    for counter in range(pd_df.shape[0]):

        ## assigning vars
        var1 = pd_df['Eop viscosity'][counter]
        var2 = pd_df['Hypo viscosity'][counter]
        var3 = pd_df['Hypo-1 Solution flow'][counter]
        var4 = pd_df['Viscosity'][counter]


        ## ML output function start
        def ML_model(output):
            data = {'Eop viscosity': [var1], 'Hypo viscosity': [var2], 'Hypo-1 Solution flow': [output]}  
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
        max_val_flow = pd_df["Hypo-1 Solution flow"].max() - std_dev_flow/2
        min_val_flow = pd_df["Hypo-1 Solution flow"].min() + std_dev_flow/2
        min_val_flow = max(0, min_val_flow)


        ## fitness func start
        desired_viscosity = 450
        def fitness_func(output):
            ####
            pred = ML_model(output)
            ####
            fitness = 100.0 / (np.abs(desired_viscosity - pred)+0.01)
            return fitness
        ## fitness func start


        ## generating children start
        def offsprings(par):
            left = max(min_val_flow, par-std_dev_flow/2)
            right = min(max_val_flow, par+std_dev_flow/2)
            child = random.uniform(left, right)		
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
                if generation == 7:
                    break
                ####
                generation = generation+1

            temp_df = pd.DataFrame({'Eop viscosity': [var1], 'Hypo viscosity': [var2], 'Actual Hypo-1 Solution flow': [var3], 'Actual Viscosity': [var4], 'Predicted Hypo -1 solution  flow': [population[-1]], 'Predicted Viscosity': [ML_model(population[-1])]})
            global dataframe1
            dataframe1 = pd.concat([dataframe1, temp_df])

        main()


    h2o.cluster().shutdown()
    return dataframe1








