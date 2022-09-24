import numpy as np
import pandas as pd
import argparse 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def crop_model(args):

    df=pd.read_csv(args.crop_csv)
    c=df.label.astype('category')
    targets = dict(enumerate(c.cat.categories))
    df['target']=c.cat.codes
    y=df.target
    X=df[['N','P','K','temperature','humidity','ph','rainfall']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    print(model.score(X_test,y_test))

    #result = model.predict([[78, 42, 42,20.130175, 81.604873, 7.628473, 262.717340]])
    #print(targets[result[0]])
    
    return model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--crop-csv", type=str)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--verbose", type=int, default=0)

    args = parser.parse_args()

    return args

if __name__ == "main":
    args = parse_args()
    crop_model(args)