import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv("water_potability.csv")
data.head()
data = data.dropna()  ###dropping null value
data.isnull().sum()   ###sum of null values

plt.figure(figsize=(5, 5))
sns.countplot(data=data, x='Potability')
plt.title("Distribution of Unsafe and Safe Water")
plt.show()

#between 6.5 8.5
import plotly.express as px
data = data
figure = px.histogram(data, x = "ph",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: PH")
figure.show()

#between 120 200
figure = px.histogram(data, x = "Hardness",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()

figure = px.histogram(data, x = "Solids",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Solids")
figure.show()

figure = px.histogram(data, x = "Sulfate",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()

figure = px.histogram(data, x = "Conductivity",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()

figure = px.histogram(data, x = "Organic_carbon",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()


figure = px.histogram(data, x = "Trihalomethanes",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()

figure = px.histogram(data, x = "Turbidity",
                      color = "Potability",
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()

correlation = data.corr()
correlation["ph"].sort_values(ascending=False)

from pycaret.classification import *
clf = setup(data, target = "Potability", session_id = 786)
compare_models()

 model = create_model("et")
predict = predict_model(model, data=data)
predict.head(10)
