import data
import model

df = data.analyze_and_explore_data()

splitData = data.split(df)

model.run_model_KNN(df, splitData)

model.run_model_random_forest(df, splitData)

model.run_model_adaboost(df,splitData)
