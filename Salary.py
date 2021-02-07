# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

Salary_Data = pd.read_csv("D:\Modules\Module 6 - SLR/Salary_Data.csv")
Salary_Data.csv

Salary_Data.columns="YE","Salary"
Salary_Data.describe()
import matplotlib.pylab as plt #for different types of plots
#first moment business decision
Salary_Data.YE.mean() # '$' is used to refer to the variables within object
Salary_Data.YE.median()
Salary_Data.YE.mode()

Salary_Data.Salary.mean() # '$' is used to refer to the variables within object
Salary_Data.Salary.median()
Salary_Data.Salary.mode()


# Measures of Dispersion / Second moment business decision
Salary_Data.YE.var() # variance
Salary_Data.YE.std()#standard deviation
Salary_Data.Salary.var()
Salary_Data.Salary.std()
range = max(Salary_Data.YE) - min(Salary_Data.YE) # range
range
range = max(Salary_Data.Salary) - min(Salary_Data.Salary) # range
range

#Third moment business decision
Salary_Data.YE.skew() # positive
Salary_Data.Salary.skew() # positive

#Fourth moment business decision
Salary_Data.YE.kurt() # positive
Salary_Data.Salary.kurt() # negative

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
Salary_Data.plot(kind = 'bar')
Salary_Data.iloc[0].plot(kind='bar')
Salary_Data.YE.plot(kind = 'bar')
Salary_Data.Salary.plot(kind = 'bar')
plt.hist(Salary_Data.YE) #histogram
plt.hist(Salary_Data.Salary)
plt.boxplot(Salary_Data.YE) #boxplot
plt.boxplot(Salary_Data.Salary)
#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(Salary_Data.YE, dist="norm",plot=pylab)
#transformation to make workex variable normal
import numpy as np
stats.probplot(Salary_Data.Salary,dist="norm",plot=pylab)



# Scatter plot
plt.scatter(x=Salary_Data['YE'], y=Salary_Data['Salary'],color='green')# positive correlation


np.corrcoef(Salary_Data.YE,Salary_Data.Salary) #strong correlation

Salary_Data.isna().sum() # no NA values

import statsmodels.formula.api as smf

# modelbuilding
model1 = smf.ols('Salary ~ YE', data=Salary_Data).fit()
model1.summary()

pred1 = model1.predict(Salary_Data)
pred1
print (model1.conf_int(0.01)) 

res1 = Salary_Data.Salary - pred1
res1
sqres1 = res1*res1
mse1 = np.mean(sqres1)
rmse1 = np.sqrt(mse1)
rmse1

plt.plot(Salary_Data['Salary'],Salary_Data['YE'])

######### Model building on Transformed Data

# Log Transformation

plt.scatter(x=np.log(Salary_Data.YE),y=Salary_Data.Salary,color='brown')
np.corrcoef(np.log(Salary_Data.YE), Salary_Data.Salary) #correlation

model2 = smf.ols('Salary ~ np.log(YE)',data=Salary_Data).fit()
model2.summary()

pred2 = model2.predict(Salary_Data)
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = Salary_Data.Salary - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=Salary_Data['YE'], y=np.log(Salary_Data['Salary']),color='orange')
#plt.show(x=delivery_time['ST'], model2.predict(delivery_time['DT'])
np.corrcoef(Salary_Data.YE, np.log(Salary_Data.Salary)) #strong correlation

model3 = smf.ols('np.log(Salary) ~ YE',data=Salary_Data).fit()
model3.summary()
model3.params
pred3 = model3.predict(Salary_Data)
pred3
pred_3 = np.exp(pred3)
pred_3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = Salary_Data.Salary - pred_3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

#polynomial 2 degree
from sklearn.preprocessing import PolynomialFeatures

X=Salary_Data['YE'].values
Y=Salary_Data['Salary'].values
X=X.reshape(-1,1)
Y=Y.reshape(-1,1)


poly_reg = PolynomialFeatures(degree = 2, include_bias=False)  #trying to create a 2 degree polynomial equation. It simply squares the x as shown in the output
X_poly = poly_reg.fit_transform(X)
print(X_poly)
poly_reg.fit(X_poly, Y)
# doing the actual polynomial Regression
from sklearn.linear_model import LinearRegression
model4 = LinearRegression()
model4.fit(X_poly, Y)

 
y_pred=model4.predict(X_poly)
y_pred

plt.scatter(X, Y, color = 'red')
plt.plot(X, model4.predict(X_poly), color = 'blue')
plt.title('polynomial regression')
plt.xlabel('YE')
plt.ylabel('Salary')
plt.show()


r_sq=model4.score(X_poly,Y)
print('cofficient of determination r_sq:',r_sq)

#polynomial 3 degree
poly_reg = PolynomialFeatures(degree = 3, include_bias=False)  #trying to create a 2 degree polynomial equation. It simply squares the x as shown in the output
X_poly = poly_reg.fit_transform(X)
print(X_poly)
poly_reg.fit(X_poly, Y)
# doing the actual polynomial Regression
from sklearn.linear_model import LinearRegression
model5= LinearRegression()
model5.fit(X_poly, Y)

 
y_pred=model5.predict(X_poly)
y_pred

r_sq=model5.score(X_poly,Y)
print('cofficient of determination r_sq:',r_sq)

plt.scatter(X, Y, color = 'red')
plt.plot(X, model5.predict(X_poly), color = 'blue')
plt.title('polynomial regression')
plt.xlabel('YE')
plt.ylabel('Salary')
plt.show()



