# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

delivery_time = pd.read_csv("D:\Modules\Module 6 - SLR/delivery_time.csv")
delivery_time
delivery_time.columns="DT","ST"
delivery_time.describe()
import matplotlib.pylab as plt #for different types of plots
#first moment business decision
delivery_time.DT.mean() # '$' is used to refer to the variables within object
delivery_time.DT.median()
delivery_time.DT.mode()

delivery_time.ST.mean() # '$' is used to refer to the variables within object
delivery_time.ST.median()
delivery_time.ST.mode()


# Measures of Dispersion / Second moment business decision
delivery_time.DT.var() # variance
delivery_time.DT.std()#standard deviation
delivery_time.ST.var()
delivery_time.ST.std()
range = max(delivery_time.DT) - min(delivery_time.DT) # range
range
range = max(delivery_time.ST) - min(delivery_time.ST) # range
range

#Third moment business decision
delivery_time.DT.skew()
delivery_time.ST.skew()

#Fourth moment business decision
delivery_time.DT.kurt()
delivery_time.ST.kurt()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np

delivery_time.DT.plot(kind = 'bar')
delivery_time.ST.plot(kind = 'bar')
plt.hist(delivery_time.DT) #histogram
plt.hist(delivery_time.ST)
plt.boxplot(delivery_time.DT) #boxplot
plt.boxplot(delivery_time.ST)
#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(delivery_time.DT, dist="norm",plot=pylab)
#transformation to make workex variable normal
import numpy as np
stats.probplot(delivery_time.ST,dist="norm",plot=pylab)
stats.probplot(np.log(delivery_time.ST),dist="norm",plot=pylab)


# Scatter plot
plt.scatter(x=delivery_time['ST'], y=delivery_time['DT'],color='green')
#possitive relation

np.corrcoef(delivery_time.DT,delivery_time.ST) #strong correlation

delivery_time.isna().sum() # no NA values

import statsmodels.formula.api as smf

model1 = smf.ols('DT ~ ST', data=delivery_time).fit()
model1.summary()

pred1 = model.predict(delivery_time)
pred1
print (model.conf_int(0.01)) # 99% confidence interval

res = delivery_time.weight - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

plt.plot(delivery_time['ST'],delivery_time['DT'])
plt.show
######### Model building on Transformed Data

# Log Transformation

plt.scatter(x=np.log(delivery_time.DT),y=delivery_time.ST,color='brown')
np.corrcoef(np.log(delivery_time.DT), delivery_time.ST) #correlation

model2 = smf.ols('DT ~ np.log(ST)',data=delivery_time).fit()
model2.summary()

pred2 = model2.predict(delivery_time)
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = delivery_time.weight - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=delivery_time['ST'], y=np.log(delivery_time['DT']),color='orange')
#plt.show(x=delivery_time['ST'], model2.predict(delivery_time['DT'])
np.corrcoef(delivery_time.ST, np.log(delivery_time.DT)) #correlation

model3 = smf.ols('np.log(DT) ~ ST',data=delivery_time).fit()
model3.summary()
model.params
pred_log = model3.predict(delivery_time)
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = delivery_time.weight - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

#polynomial 2 degree
from sklearn.preprocessing import PolynomialFeatures

X=delivery_time['ST'].values
Y=delivery_time['DT'].values
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
#plt.plot(X, Y, color = 'red')
plt.plot(X, model4.predict(X_poly), color = 'blue')
plt.title('polynomial regression')
plt.xlabel('ST')
plt.ylabel('DT')
plt.show()
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)


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

#from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)


r_sq=model5.score(X_poly,Y)
print('cofficient of determination r_sq:',r_sq)

plt.scatter(X, Y, color = 'red')
#plt.plot(X, Y, color = 'red')
plt.plot(X, model5.predict(X_poly), color = 'blue')
plt.title('polynomial regression')
plt.xlabel('ST')
plt.ylabel('DT')
plt.show()


