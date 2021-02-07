#=============================================================
#            Sol - 1
#=============================================================

library(readr)
library(readr)
calories_consumed <- read_csv("D:/Modules/Module 6 - SLR/calories_consumed.csv")
View(calories_consumed)
summary(calories_consumed)
attach(calories_consumed)
plot(calories_consumed$`Weight gained (grams)`,calories_consumed$`Calories Consumed`)# positively related

cor(`Weight gained (grams)`,`Calories Consumed`)#strong corelated r=0.94


sum(is.na(calories_consumed))###no NA vales

#EDA
#1st business moments
mean(calories_consumed$`Weight gained (grams)`) #357.71
mean(calories_consumed$`Calories Consumed`) #2340.714

median(calories_consumed$`Weight gained (grams)`) #200
median(calories_consumed$`Calories Consumed`) #2250

# 2nd business moments
var(calories_consumed$`Weight gained (grams)`) #111350.7
var(calories_consumed$`Calories Consumed`) #565668.7

sd(calories_consumed$`Weight gained (grams)`) #333.6926
sd(calories_consumed$`Calories Consumed`) #752.1095

range(calories_consumed$`Weight gained (grams)`) #62 1100
range(calories_consumed$`Calories Consumed`) #1400 3900

#different visulaizations
boxplot(calories_consumed) #no outliers
barplot(calories_consumed$`Weight gained (grams)`)
barplot(calories_consumed$`Calories Consumed`)
hist(calories_consumed$`Weight gained (grams)`)
hist(calories_consumed$`Calories Consumed`)# most of the alories consumed from 0 to 200
#3rd business moment
library(moments)
skewness(calories_consumed$`Weight gained (grams)`)#positive skewness
skewness(calories_consumed$`Calories Consumed`) #positive skewness
#4th business moment
kurtosis(calories_consumed$`Weight gained (grams)`) #positive kurtosis
kurtosis(calories_consumed$`Calories Consumed`) # positive kurtosis

#Normal Quantile-Quantile Plot
qqnorm(calories_consumed$`Weight gained (grams)`)## Normally distributed
qqline(calories_consumed$`Weight gained (grams)`)
qqnorm(calories_consumed$`Calories Consumed`)
qqline(calories_consumed$`Calories Consumed`)
#transformation to make Calories Consumed variable normal
qqnorm(log(`Calories Consumed`))
qqline(log(`Calories Consumed`)) # Normally Distributed

reg1 <- lm(`Weight gained (grams)`~`Calories Consumed`)
summary (reg1)
confint(reg1,level = 0.95)
pred1<-predict(reg1,interval="predict")
predict(reg, interval = "confidence")
Calories1<-cbind(calories_consumed,pred1)
#p value is significant and Multiple R-squared:  0.8968 is greater than 0.87 so strong corelation
ggplot(data=calories_consumed, aes(x=`Calories Consumed`, y=`Weight gained (grams)`)) + geom_point(colours='blue') + geom_line(colour='red',data=Calories1, aes(x=`Calories Consumed`, y=fit)) 

reg1$residuals
reg1$fitted.values
rmse1 <- sqrt(mean(reg1$residuals^2))
rmse1

attach(calories_consumed)

# Log model

reg2 <- lm(`Weight gained (grams)` ~ log(`Calories Consumed`)) 
summary(reg2)

confint(reg1,level = 0.95)
pred2<-predict(reg2,interval="predict")
predict(reg2, interval = "confidence")
Calories2<-cbind(calories_consumed,pred2)
ggplot(data=calories_consumed, aes(x=`Calories Consumed`, y=`Weight gained (grams)`)) + geom_point(colours='blue') + geom_line(colour='red',data=Calories2, aes(x=`Calories Consumed`, y=fit)) 

rmse2 <- sqrt(mean(reg2$residuals^2))
rmse2


# Exp model
plot(`Calories Consumed`, log(`Weight gained (grams)`))
cor(`Calories Consumed`, log(`Weight gained (grams)`)
reg3 <- lm(log(`Weight gained (grams)`)~ (`Calories Consumed`)) # exponential tranformation
summary(reg3)

confint(reg1,level = 0.95)
pred3<-predict(reg3,interval="predict")
predict(reg3, interval = "confidence")
Calories3<-cbind(calories_consumed,pred3)
ggplot(data=calories_consumed, aes(x=`Calories Consumed`, y=`Weight gained (grams)`)) + geom_point(colours='blue') + geom_line(colour='red',data=Calories3, aes(x=`Calories Consumed`, y=fit)) 

rmse3 <- sqrt(mean(reg3$residuals^2))
rmse3

#Quadratic Transformation
reg4 <- lm(`Weight gained (grams)` ~ `Calories Consumed`+ I (`Calories Consumed`^2)) 
summary(reg4)

confint(reg1,level = 0.95)
pred4<-predict(reg4,interval="predict")
predict(reg4, interval = "confidence")
Calories4<-cbind(calories_consumed,pred4)
ggplot(data=calories_consumed, aes(x=`Calories Consumed`, y=`Weight gained (grams)`)) + geom_point(colours='blue') + geom_line(colour='red',data=Calories4, aes(x=`Calories Consumed`, y=fit)) 

rmse4 <- sqrt(mean(reg4$residuals^2))
rmse4

# polynomial Transformation

reg5 <- lm(`Weight gained (grams)` ~ `Calories Consumed` + I(`Calories Consumed`^2) + I (`Calories Consumed`^3)) # log transformation
summary(reg5)

confint(reg5,level = 0.95)
pred5<-predict(reg5,interval="predict")
predict(reg5, interval = "confidence")
Calories5<-cbind(calories_consumed,pred5)
ggplot(data=calories_consumed, aes(x=`Calories Consumed`, y=`Weight gained (grams)`)) + geom_point(colours='blue') + geom_line(colour='red',data=Calories5, aes(x=`Calories Consumed`, y=fit)) 

rmse5 <- sqrt(mean(reg2$residuals^2))
rmse5
#conclusion - after applying transformation we are getting more R- sqaured value in model5 (reg5), so moldel 5 is the best model.

#=============================================================
#            Sol - 2
#=============================================================

library(moments)
library(readr)
delivery_time <- read_csv("D:/Modules/Module 6 - SLR/delivery_time.csv")
View(delivery_time)
attach(delivery_time)
summary(delivery_time)
#EDA
#1st business moments
mean (delivery_time$`Delivery Time`) #16.790
mean(delivery_time$`Sorting Time`) #6.190

median (delivery_time$`Delivery Time`) #17.83
median(delivery_time$`Sorting Time`)#6

# 2nd business moments
var (delivery_time$`Delivery Time`) #25.754
var(delivery_time$`Sorting Time`) #6.46

sd (delivery_time$`Delivery Time`) #5.074
sd(delivery_time$`Sorting Time`) #2.54

range (delivery_time$`Delivery Time`) #8 29
range(delivery_time$`Sorting Time`) #2 10

#outliers
sum(is.na(delivery_time)) #no NA vales

#different visulaizations
boxplot(delivery_time) #no outliers
barplot(delivery_time)
hist(delivery_time$`Delivery Time`)
hist(delivery_time$`Sorting Time`)
#3rd business moment
library(moments)
skewness(delivery_time$`Delivery Time`)#positive skewness
skewness(delivery_time$`Sorting Time`) #positive skewness
#4th business moment
kurtosis(delivery_time$`Delivery Time`) #positive kurtosis
kurtosis(delivery_time$`Sorting Time`) # positive kurtosis

#Normal Quantile-Quantile Plot
qqnorm(delivery_time$`Delivery Time`)## Normally distributed
qqline(delivery_time$`Delivery Time`)
qqnorm(delivery_time$`Sorting Time`)
qqline(delivery_time$`Sorting Time`)
#transformation to make Sorting Time variable normal
qqnorm(log(`Sorting Time`))
qqline(log(`Sorting Time`)) # Normally Distributed

plot(`Delivery Time`,`Sorting Time`)
cor(`Delivery Time`,`Sorting Time`)#0.82 moderate corelation



#model building
model1<-lm(`Delivery Time`~`Sorting Time`)
summary(model1)

confint(model2,level=0.95)
pred1<-predict(model1,interval="predict")

delivery_time1<-cbind(delivery_time,pred1)
ggplot(data=delivery_time, aes(x=`Sorting Time`, y=`Delivery Time`)) + geom_point(colours='blue') + geom_line(colour='red',data=delivery_time1, aes(x=`Sorting Time`, y=fit)) 

model1$residuals
rmse1<-sqrt(mean(model1$residuals^2))
rmse1

#p is significant but r value is less 0.68
#R squared value=0.68 need to do transformation

# log transformation
model2 <- lm((`Delivery Time`)~log(`Sorting Time`))
summary(model2)
#Now R squared increased to value=0.71
confint(model2,level=0.95)
pred2<-predict(model2,interval="predict")

delivery_time2<-cbind(delivery_time,pred2)
ggplot(data=delivery_time, aes(x=`Sorting Time`, y=`Delivery Time`)) + geom_point(colours='blue') + geom_line(colour='red',data=delivery_time2, aes(x=`Sorting Time`, y=fit)) 

model2$residuals
rmse2<-sqrt(mean(model2$residuals^2))
rmse2

# Exponential transformation
model3 <- lm(log(`Delivery Time`)~ `Sorting Time`) # exponential tranformation
summary(model3)

confint(reg1,level = 0.95)
pred3<-predict(model3,interval="predict")
predict(model3, interval = "confidence")
delivery_time3<-cbind(delivery_time,pred3)
ggplot(data=delivery_time, aes(x=`Sorting Time`, y=`Delivery Time`)) + geom_point(colours='blue') + geom_line(colour='red',data=delivery_time3, aes(x=`Sorting Time`, y=fit)) 

rmse3 <- sqrt(mean(reg3$residuals^2))
rmse3

#Quadratic Transformation
model4 <- lm(`Delivery Time`~ `Sorting Time`+ I(`Sorting Time`^2)) # exponential tranformation
summary(model4)

confint(reg4,level = 0.95)
pred4<-predict(model4,interval="predict")
predict(model4, interval = "confidence")
delivery_time4<-cbind(delivery_time,pred4)
ggplot(data=delivery_time, aes(x=`Sorting Time`, y=`Delivery Time`)) + geom_point(colours='blue') + geom_line(colour='red',data=delivery_time4, aes(x=`Sorting Time`, y=fit)) 

rmse4 <- sqrt(mean(model3$residuals^2))
rmse4

# polynomial Transformation

reg5 <- lm(`Weight gained (grams)` ~ `Calories Consumed` + I(`Calories Consumed`^2) + I (`Calories Consumed`^3)) # log transformation
summary(reg5)

model5 <- lm(`Delivery Time`~ `Sorting Time`+ I(`Sorting Time`^2)+I(`Sorting Time`^3)) # exponential tranformation
summary(model5)

confint(reg1,level = 0.95)
pred5<-predict(model5,interval="predict")
predict(model5, interval = "confidence")
delivery_time5<-cbind(delivery_time,pred5)
ggplot(data=delivery_time, aes(x=`Sorting Time`, y=`Delivery Time`)) + geom_point(colours='blue') + geom_line(colour='red',data=delivery_time5, aes(x=`Sorting Time`, y=fit)) 

rmse5 <- sqrt(mean(reg3$residuals^2))
rmse5


##########sol 3#############

emp_data <- read_csv("D:/Modules/Module 6 - SLR/emp_data.csv")
View(emp_data)
summary(emp_data)
#EDA
#1st business moments
mean (emp_data$Salary_hike) #1688.6
mean(emp_data$Churn_out_rate) #72.9

median (emp_data$Salary_hike) #1675
median(emp_data$Churn_out_rate)#71

# 2nd business moments
var (emp_data$Salary_hike) #8481.822
var(emp_data$Churn_out_rate) #105.2111

sd (emp_data$Salary_hike) #92.09
sd(emp_data$Churn_out_rate) #10.25

range (emp_data$Salary_hike) #1580 1870
range(emp_data$Churn_out_rate) #60 90

#outliers
sum(is.na(emp_data)) #no NA vales
summary(emp_data)
#different visulaizations
boxplot(emp_data) #no outliers
barplot(emp_data$Salary_hike)
barplot(emp_data$Churn_out_rate)
hist(emp_data$Salary_hike)  ##maximum data are from 1550 to 1650
hist(emp_data$Churn_out_rate)
#3rd business moment
library(moments)
skewness(emp_data$Salary_hike)#positive skewness
skewness(emp_data$Churn_out_rate) #positive skewness
#4th business moment
kurtosis(emp_data$Salary_hike) #positive kurtosis
kurtosis(emp_data$Churn_out_rate) # positive kurtosis

#Normal Quantile-Quantile Plot
qqnorm(emp_data$Salary_hike)## Normally distributed
qqline(emp_data$Salary_hike)
qqnorm(emp_data$Churn_out_rate)##Normally distributed
qqline(emp_data$Churn_out_rate)
attach(emp_data)

plot(Salary_hike,Churn_out_rate)##negative realtion
cor(Salary_hike,Churn_out_rate)#-0.911 strong negative corelation

#model building
model1<- lm(Churn_out_rate~Salary_hike)
summary (model1) #r=0.83

confint(reg1,level = 0.95)
pred1<-predict(model1,interval="predict")
predict(model1, interval = "confidence")
emp_data1<-cbind(emp_data,pred1)
ggplot(data=emp_data, aes(x=Salary_hike, y=Churn_out_rate)) + geom_point(colours='blue') + geom_line(colour='red',data=emp_data1, aes(x=Salary_hike, y=fit)) 

rmse1 <- sqrt(mean(reg1$residuals^2))
rmse1

#appying transformation to get least rmse value
model2 <- lm(Churn_out_rate~log(Salary_hike))
summary(model2)

confint(reg1,level = 0.95)
pred2<-predict(model1,interval="predict")
predict(model2, interval = "confidence")
emp_data2<-cbind(emp_data,pred2)
ggplot(data=emp_data, aes(x=Salary_hike, y=Churn_out_rate)) + geom_point(colours='blue') + geom_line(colour='red',data=emp_data2, aes(x=Salary_hike, y=fit)) 

rmse2 <- sqrt(mean(model3$residuals^2))
rmse2

# Exponential Transformation
model3 <- lm(log(Churn_out_rate)~Salary_hike)
summary(model3)

confint(reg1,level = 0.95)
pred3<-predict(model3,interval="predict")
predict(model3, interval = "confidence")
emp_data3<-cbind(emp_data,pred3)
ggplot(data=emp_data, aes(x=Salary_hike, y=Churn_out_rate)) + geom_point(colours='blue') + geom_line(colour='red',data=emp_data3, aes(x=Salary_hike, y=fit)) 

rmse3 <- sqrt(mean(model3$residuals^2))
rmse3

# Quadratic Transformation
model4 <- lm(Churn_out_rate~Salary_hike + I(Salary_hike^2))
summary(model4)

confint(reg1,level = 0.95)
pred4<-predict(model4,interval="predict")
predict(model4, interval = "confidence")
emp_data4<-cbind(emp_data,pred4)
ggplot(data=emp_data, aes(x=Salary_hike, y=Churn_out_rate)) + geom_point(colours='blue') + geom_line(colour='red',data=emp_data4, aes(x=Salary_hike, y=fit)) 

rmse4 <- sqrt(mean(model4$residuals^2))
rmse4

#Polynomial transformation

model5 <- lm(Churn_out_rate~Salary_hike + I(Salary_hike^2)+I(Salary_hike^3))
summary(model5)

confint(reg1,level = 0.95)
pred5<-predict(model5,interval="predict")
predict(model5, interval = "confidence")
emp_data5<-cbind(emp_data,pred5)
ggplot(data=emp_data, aes(x=Salary_hike, y=Churn_out_rate)) + geom_point(colours='blue') + geom_line(colour='red',data=emp_data5, aes(x=Salary_hike, y=fit)) 

rmse4 <- sqrt(mean(model4$residuals^2))
rmse4
###########sol 4###########

library(readr)
Salary_Data <- read_csv("D:/Modules/Module 6 - SLR/Salary_Data.csv")
View(Salary_Data)
attach(Salary_Data)
summary(Salary_Data)
#EDA
#1st business moments
mean(YearsExperience) #5.31
mean(Salary_Data$Salary) #76003

median(Salary_Data$YearsExperience) #4.7
median(Salary_Data$Salary) #65237

# 2nd business moments
var(Salary_Data$YearsExperience) #8.053
var(Salary_Data$Salary)

sd(Salary_Data$YearsExperience) #2.83
sd(Salary_Data$Salary) #27414.43

range(Salary_Data$YearsExperience) #1.1 10.5
range(Salary_Data$Salary) #37731 122391

#different visulaizations
boxplot(Salary_Data) #no outliers

hist(Salary_Data$YearsExperience)
hist(Salary_Data$Salary)
#3rd business moment
library(moments)
skewness(Salary_Data$YearsExperience) #positive skewness
skewness(Salary_Data$Salary) #positive skewness
#4th business moment
kurtosis(Salary_Data$YearsExperience) #positive kurtosis
kurtosis(Salary_Data$Salary) # positive kurtosis

#Normal Quantile-Quantile Plot
qqnorm(Salary)
qqline(Salary)
qqnorm(YearsExperience)##Normally distributed
qqline(YearsExperience)
attach(Salary_Data)
plot(YearsExperience,Salary) #positive relation

cor(YearsExperience,Salary) #strongly corelated r=0.97



#model building
model1<- lm(Salary~YearsExperience)
summary (model1)
#p is significant and r=0.95
confint(model1,level = 0.95)
pred1<-predict(model1,interval="predict")
Salary_Data1<-cbind(Salary_Data,pred1)
Salary_Data1
model1$residuals
rmse<-sqrt(mean(model1$residuals^2))
rmse  
#least RMSE value model is evualted
library(ggplot2)
ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_point(colours='blue') + geom_line(colour='red',data=Salary_Data1, aes(x=YearsExperience, y=fit)) 
#ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_line(colour='red',size=100,alpha=0.4) 
#ggplot(data=Salary_Data)+ geom_point(mapping = aes(x=YearsExperience, y=Salary,colours='blue')) + geom_line(colour='red',data=Salary_Data, aes(x=YearsExperience, y=predict)) 

# logarithmic transformation
model2<- lm(Salary~log(YearsExperience))
summary(model2)
confint(model2,level=0.95)
pred2<-predict(model2,interval="predict")
ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_point(colours='blue') + geom_line(colour='red',data=Salary_Data2, aes(x=YearsExperience, y=fit)) 

Salary_Data2<-cbind(Salary_Data,pred2)
model2$residuals
rmse2<-sqrt(mean(model2$residuals^2))
rmse2
# Exponenetial transformation
model3<- lm(log(Salary)~YearsExperience)
summary(model3)

confint(model3,level=0.95)
pred3<-predict(model3,interval="predict")
Salary_Data3<-cbind(Salary_Data,pred3)
ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_point(colours='blue') + geom_line(colour='red',data=Salary_Data3, aes(x=YearsExperience, y=fit)) 

model3$residuals
rmse3<-sqrt(mean(model3$residuals^2))
rmse3

# polynomial 2D degree
#model4<- lm(log(Salary)~YearsExperience + I(YearsExperience^2))
model4<- lm(Salary~YearsExperience + I(YearsExperience^2))
summary(model4)

confint(model4,level=0.95)
pred4<-predict(model4,interval="predict")
Salary_Data4<-cbind(Salary_Data,pred4)
ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_point(colours='blue') + geom_line(colour='red',data=Salary_Data4, aes(x=YearsExperience, y=fit)) 

model4$residuals
rmse4<-sqrt(mean(model4$residuals^2))
rmse4

# polynominal 3D transformation
model5<- lm(Salary~YearsExperience + I(YearsExperience^2)+I(YearsExperience^3))
summary(model5)
confint(model5,level=0.95)
pred5<-predict(model5,interval="predict")
Salary_Data5<-cbind(Salary_Data,pred5)
ggplot(data=Salary_Data, aes(x=YearsExperience, y=Salary)) + geom_point(colours='blue') + geom_line(colour='red',data=Salary_Data5, aes(x=YearsExperience, y=fit)) 

model$residuals
rmse5<-sqrt(mean(model5$residuals^2))
rmse5
#

