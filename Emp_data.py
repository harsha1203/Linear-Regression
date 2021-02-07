# For reading data set
# importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

emp_data = pd.read_csv("D:\Modules\Module 6 - SLR/emp_data.csv")
emp_data.csv

emp_data.columns="SH","COR"
emp_data.describe()
import matplotlib.pylab as plt #for different types of plots
#first moment business decision
emp_data.SH.mean() # '$' is used to refer to the variables within object
emp_data.SH.median()
emp_data.SH.mode()

emp_data.COR.mean() # '$' is used to refer to the variables within object
emp_data.COR.median()
emp_data.COR.mode()


# Measures of Dispersion / Second moment business decision
emp_data.SH.var() # variance
emp_data.SH.std()#standard deviation
emp_data.COR.var()
emp_data.COR.std()
range = max(emp_data.SH) - min(emp_data.SH) # range
range
range = max(emp_data.COR) - min(emp_data.COR) # range
range

#Third moment business decision
emp_data.SH.skew() # positive
emp_data.COR.skew() # positive

#Fourth moment business decision
emp_data.SH.kurt() # positive
emp_data.COR.kurt() # negative

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np

emp_data.SH.plot(kind = 'bar')
emp_data.COR.plot(kind = 'bar')
plt.hist(emp_data.SH) #histogram
plt.hist(emp_data.COR)
plt.boxplot(emp_data.SH) #boxplot
plt.boxplot(emp_data.COR)
#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(emp_data.SH, dist="norm",plot=pylab)
#transformation to make workex variable normal
import numpy as np
stats.probplot(emp_data.COR,dist="norm",plot=pylab)



# Scatter plot
plt.scatter(x=emp_data['SH'], y=emp_data['COR'],color='green')
#negative relation

np.corrcoef(emp_data.SH,emp_data.COR) #strong negative correlation

emp_data.isna().sum() # no NA values

import statsmodels.formula.api as smf

# modelbuilding
model1 = smf.ols('COR ~ SH', data=emp_data).fit()
model1.summary()

pred1 = model1.predict(emp_data)
pred1
print (model1.conf_int(0.01)) 

res = emp_data.COR - pred1
res
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse)
rmse

plt.plot(emp_data['COR'],emp_data['SH'])
plt.show
######### Model building on Transformed Data

# Log Transformation

plt.scatter(x=np.log(emp_data.SH),y=emp_data.COR,color='brown')
np.corrcoef(np.log(emp_data.SH), emp_data.COR) #correlation

model2 = smf.ols('COR ~ np.log(SH)',data=emp_data).fit()
model2.summary()

pred2 = model2.predict(emp_data)
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = emp_data.COR - pred2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=emp_data['SH'], y=np.log(emp_data['COR']),color='orange')
#plt.show(x=delivery_time['ST'], model2.predict(delivery_time['DT'])
np.corrcoef(emp_data.SH, np.log(emp_data.COR)) #correlation

model3 = smf.ols('np.log(COR) ~ SH',data=emp_data).fit()
model3.summary()
model3.params
pred3 = model3.predict(emp_data)
pred3
pred_3 = np.exp(pred3)
pred_3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = emp_data.COR - pred_3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

#polynomial 2 degree
from sklearn.preprocessing import PolynomialFeatures

X=emp_data['SH'].values
Y=emp_data['COR'].values
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
plt.xlabel('SH')
plt.ylabel('COR')
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

#from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)


r_sq=model5.score(X_poly,Y)
print('cofficient of determination r_sq:',r_sq)

plt.scatter(X, Y, color = 'red')
plt.plot(X, model5.predict(X_poly), color = 'blue')
plt.title('polynomial regression')
plt.xlabel('SH')
plt.ylabel('COR')
plt.show()


