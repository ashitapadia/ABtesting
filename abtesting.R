#### 1. Import libraries and set working directory ####
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(pwr)
library(plyr)
setwd("C:\\Users\\srivastavas\\Desktop\\TM\\")

#### 2. Read in files and merge ####
eng_df<-read.csv("email_engagement.csv")
length(unique(eng_df$user_id))
dim(eng_df) #all user ids are unique with no duplicates - 4,648 ids

test_conv_df<-read.csv("test_conversions.csv") #those that converted as a result of A/B test

length(unique(test_conv_df$user_id)) #only 2,210 ids
summary(test_conv_df) #only contains conversion data
variations_df<-read.csv("variations.csv")
visits_df<-read.csv("visits.csv")
length(unique(visits_df$user_id)) #more user_ids than in the email engagement file


merge_1<-merge(variations_df,visits_df,by.x="user_id",by.y="user_id")
merge_2<-merge(merge_1,test_conv_df,by.x="user_id",by.y="user_id",all.x=TRUE)
merge_3<-merge(merge_2,eng_df,by.x="user_id",by.y="user_id",all.x=TRUE)

#Assume that if NA for conversion then 0. i.e. user did not click
#Assume that if NA for clicked then 0. i.e. user did not click
merge_3$converted<-if_else(is.na(merge_3$converted),0,1)
merge_3$clicked_on_email<-if_else(is.na(merge_3$clicked_on_email),0,1)
merge_3$converted<-as.factor(merge_3$converted)
merge_3$clicked_on_email<-as.factor(merge_3$clicked_on_email)

#### 3. Data Transformation ####
#convert visit time into a useful variable
merge_3$timeofday<-   mapvalues(hour(merge_3$visit_time),from=c(0:23),
                           to=c(rep("night",times=5), rep("morning",times=6),rep("afternoon",times=5),rep("night", times=8)))

merge_3$timeofday<-as.factor(merge_3$timeofday)

#### 4. Exploratory data analysis to understand users ####
#Income ranges from 14,000 to $1511958 - too much variation to see a trend

l <- ggplot(merge_3, aes(variation,fill = converted))
l <- l + geom_histogram(stat="count")
print(l)
print("Proportion that converted by variation converted")
tapply(as.numeric(merge_3$converted) - 1 ,merge_3$variation,mean)
#slightly greater proportion of those that viewed treatment variation converted


### 5.1 A/B testing - visits, variations and test converions
# Null Hypothesis: Assumption that there is no difference between the conversion rates for control & exp
#Alternative Hypothesis: There is a difference between the conversion rates for control & exp
control_sz<-length(which(merge_3$variation=="control"))
exp_sz<-length(which(merge_3$variation=="treatment"))
control_yes<-length(which(merge_3$variation=="control" & merge_3$converted=="1"))
exp_yes<-length(which(merge_3$variation=="treatment" & merge_3$converted=="1"))
prop.test(c (control_yes, exp_yes), c (control_sz, exp_sz))
#p-value<0.05 - so yes there is a difference in conversion rates. Slightly higher for treatment

#chi-sq test for conversion and variation
#Null hypothesis: There is no association between conversion and variation

ch_test<-chisq.test(merge_3$variation,merge_3$converted) #p-value < 0.05, There may be an association between
#likelihood of conversion and variation
ch_test$stdres
#Pearson's std residuals measure
#  how large is the deviation from each cell to the null hypothesis (in this case, independence between row and column's).
#treatment is negatively associated with non-conversion, so it has fewer proportion of people who did not convert


#test which variables are key in determining conversion via logistic regression
## 5.2 Further analysis - logistic regression ##
glm_model<-glm(converted~variation+channel+age+gender+timeofday+income,data=merge_3,family = binomial(link="logit"))
summary(glm_model)
#only variation treatment is statistically signficant
#positive coefficient indicates that all other factors remaining fixed, if person has viewed treatment variation, they are more likely to convert
#viewing treatment rather than control increases odds by 0.0272 for conversion

anova(glm_model, test="Chisq")
#Finding: Only adding channel and income is statistically significant in reducing the residual deviance.
#How our model does against model without intercep

#assess model fit
library(pscl)
pR2(glm_model) #McFadden R2 index can be used to assess the model fit.
#only 12.94$ of variation in whether or not someone will convert is likely to be explained by current model

#### 5. Exploratory learning for engagemnt with emails  ####

#Exploratory data analysis to understand relationship between attributes and engagement with emails

l <- ggplot(merge_3, aes(gender,fill = clicked_on_email))
l <- l + geom_histogram(stat="count")
print(l)
print("Greater proportion of males clicked on email")
tapply(as.numeric(merge_3$clicked_on_email) - 1 ,merge_3$gender,mean)

l <- ggplot(merge_3, aes(timeofday,fill = clicked_on_email))
l <- l + geom_histogram(stat="count")
print(l)
print("Greatest proportion of people that visited website visited in the nighttime. Similar proportions of those that clicked on email by time of day")
tapply(as.numeric(merge_3$clicked_on_email) - 1 ,merge_3$timeofday,mean)


l <- ggplot(merge_3, aes(channel,fill = clicked_on_email))
l <- l + geom_histogram(stat="count")
print(l)
print("Those that had PPC as their channel had the highest proportion of clicking on email (73%).
      However, highest number of users were prompted to visit the website due to TV")
tapply(as.numeric(merge_3$clicked_on_email) - 1 ,merge_3$channel,mean)


#Age has 1,243 missing values
row.has.na <- apply(merge_3, 1, function(x){any(is.na(x))})
sum(row.has.na)
final.filtered <- merge_3[!row.has.na,]

ggplot(final.filtered,aes(x=clicked_on_email,y=age))+geom_boxplot()
print("On average, those that converted were on average 38 years old, while those that didn't were around 32 years of age")

dplyr::group_by(final.filtered, clicked_on_email) %>%
  dplyr::summarise(mean=mean(age), sd=sd(age))


ggplot(final.filtered,aes(x=clicked_on_email,y=income))+geom_boxplot()
dplyr::group_by(final.filtered, clicked_on_email) %>%
  dplyr::summarise(mean=mean(income), sd=sd(income))
print("On average, those that converted on average earned more than those that didn't. High variation in income, with a lot of very high income values in both groups")


#### 6. Machine learning model for engagement with emails ####
library(caTools)
library(e1071)
library(glmnet)

## drop columns not required
mydatanew = final.filtered[,-c(1,3,8)]
#randomly split data into training, test and validation
#Splitting data
split <- sample.split(mydatanew$clicked_on_email, SplitRatio = 0.70) 
train <- subset(mydatanew, split == T) #ensure same proportion of clicks and non-clicks in both training and test sets to ensure balance
test <- subset(mydatanew, split == F)


## Testing model accuracy on test set
model_glm <- glm(clicked_on_email ~ ., data = train, family='binomial') 
predicted_glm <- predict(model_glm, test, type='response')
predicted_glm <- ifelse(predicted_glm > 0.5,1,0)
misClasificError <- mean(predicted_glm != test$clicked_on_email)
print(paste('Accuracy',1-misClasificError))


print("The 0.84 accuracy on the test set is quite a good result. However, keep in mind that this result is somewhat dependent on the manual split of the data that I made earlier, therefore if you wish for a more precise score, you would be better off running some kind of cross validation such as k-fold cross validation.
      As a last step, we are going to plot the ROC curve and calculate the AUC (area under the curve) which are typical performance measurements for a binary classifier.
      The ROC is a curve generated by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings while the AUC is the area under the ROC curve. As a rule of thumb, a model with good predictive ability should have an AUC closer to 1 (1 is ideal) than to 0.5.")
library(ROCR)
p<-predicted_glm
pr <- prediction(p, test$clicked_on_email)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


summary(model_glm)
print("Statistically signficant predicturs are channel, age, and gender where males are 
more likely to click on email and those that come via Facebook, followed by PPC and then TV. Also, an older user is more likely to click on an email than a younger user")

##Response to Q1: There is a statistically significant difference in those that converted based on variation; however, the difference in proportion is not very large. I would recommend re-running the test with a larger dataset to ensure reproducibility. 

##Response to Q2: The model can predict whether or not someone will click on an email with 84% accuracy. Top predictors are gender, age, and channel. 
