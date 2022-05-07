import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def analyze_and_explore_data():
    plt.style.use("ggplot")
    desired_width = 320
    pd.set_option('display.width', desired_width)
    df = pd.read_csv("WineQT.csv")
    print("*****  Vino verde red wine quality  *****")
    print("DATA")
    # See the number of rows and columns
    print("Rows, columns: " + str(df.shape))
    # See the first five rows of the dataset
    print("First five rows of the data set..")
    print(df.head())
    print("----------------------------------------------------------------------------------------------")

    # Missing Values
    print("Checking missing Values:")
    print(df.isna().sum())
    print("----------------------------------------------------------------------------------------------")

    print("INFO:")
    print(df.info())
    print("----------------------------------------------------------------------------------------------")

    print("Describe value data set:")
    print(df.describe().round(2))
    print("----------------------------------------------------------------------------------------------")

    # Drop columns ID , because we don't need it.
    df.drop(columns="Id", inplace=True)

    # The unique quality
    print("The Value Quality ", df["quality"].unique())
    print("----------------------------------------------------------------------------------------------")

    # graph all the data set - just for looking
    ### df.plot(figsize=(15, 7))

    print("Making Group by '"'quality'"' : ")
    ave_qu = df.groupby("quality").mean()
    print(ave_qu)
    # graph the group by
    ### ave_qu.plot(kind="bar",figsize=(20,10))
    print("----------------------------------------------------------------------------------------------")

    return df


def split(df):
    # Separate feature variables and target variable
    X = df.drop(['quality'], axis=1)
    y = df['quality']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Splitting the data - train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print("X Train : ", x_train.shape)
    print("X Test  : ", x_test.shape)
    print("Y Train : ", y_train.shape)
    print("Y Test  : ", y_test.shape)
    return [x_train, x_test, y_train, y_test]
