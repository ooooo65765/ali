# Classifying/Predicting  Molecular Subtypes in Breast Cancer Tumors 

This data set contains information on breast cancer patients with 4 major molecular subtypes of tumors: Luminal A, Luminal B, HER2 Enriched, and Basal(triple negative).  

#### Methods used for classifying molecular subtypes are Ensemble-Boosting, Multinomial,K Nearest Neighbors,Support Vector Machine,  Naïve Bayes, eXtreme Gradient Boosting, and Neural Network 

## Table of Contents

#### Part 1: Introduction
#### Part 2: Information on Dataset
#### Part 3: Loading Packages and Reading in datasets in Rstudio
#### Part 4: Using Stringr to Manipulate Complete TCGA ID
#### Part 5:  Using Only Variables/Predictors With At Least 99% Of The Data
#### Part 6: Using Lasso For Variable/SubSet Selection.
#### Part 7: Retrieving  Indexes of Relevant Variables From Lasso 
#### Part 8: Selecting Predictors By Column Number 
#### Part 9: Classifying Molecular Tumor Type Using Ensemble, Multinomial, K Nearest Neighbors, Support Vector Machine, and Other Machine Learning Methods
#### Part 10 : Conclusions



# Part 1: Introduction 

Breast Cancer is a multifactorial disease that forms in the cells of the breast. Breast cancer can occur in both men and women, but it's far more common in women. Breast cancer is the most common form of cancer in American women, the average risk of an American Women developing breast cancer in her life time is about 1 in 7 or about 12%. Of that 12 percent of women who do develop breast cancer, the cancer that a breast cancer patient will die from her disease is 2.7%.    

**Density map of worldwide incidents of Breast Cancer**
![breastcancerstatsworldwide](http://www.worldwidebreastcancer.com/wp-content/uploads/2011/08/breastcancerstatsworldwide.jpg)

Breast cancer is often referred to as one disease, however there are many different types of breast cancers based on varying types of tumors. Tumors can vary in location, size, shape, and grade(severity). These characteristics, along with hormone receptor status and HER2 status affect prognosis. 

**Breast Cancer by the Numbers** 
![bythenumbers](http://www.julep.com/blog/wp-content/uploads/2014/10/FINAL-2014-Infographic-Julep_EdithSanford_5x5-1.jpg)

There are many factors that increase one’s chances in developing breast cancer, mutations in DNA are often the root causes of breast cancer. Other causes that may influence one’s susceptibility to developing breast cancer are mutations to genes that assist in cell differentiation (Proto-Oncogenes), and mutations to genes that modulate cell division, cellular repair, apoptosis (Tumor Suppressor Genes).  

Breast cancer awareness and research funding has helped created advances in the diagnosis and treatment of breast cancer. Medical advances and innovation in treating breast cancer has increased survival rates, lower remission rates, and lower the number of deaths associated with the disease. More recently, the introduction of precision medicine and gene therapy has the potential to transform how physician treat breast cancer. Precision medicine refers to the tailoring of medical treatment based on the cellular profile of a disease and the patient’s genome. 

Research has identified 4 major molecular subtypes within breast cancer tumors, these subtypes are based on the genes that cancer cell express: Luminal A, Luminal B, HER2 Enriched, and Basal (triple negative). Identifying and studying these subtypes has potential in planning more effective treatment and developing new therapies. Currently, Prognosis and treatment decisions are guided mainly by tumor stage, tumor grade, hormone receptor status and HER2 status. Molecular subtypes are mostly used in research settings; they are not part of a patients report and are not used to guide treatment. However, the use of molecular subtypes has greatly expanded, based on determining what genes are expressed in tumor samples, identifying subtypes of tumors can improve prognosis.

# **The scope of this project is to correctly identify the molecular subtypes of tumors based on differences in gene expression. ** 
 

# Part 2: Information on Dataset


1. 77_cancer_proteomes_CPTAC_itraq.csv

	Includes protein expression values and metadata on 12553 genes from 80 breast cancer patients and 3 healthy individuals.


2. clinical_data_breast_cancer.csv

	Contains clinical data and  various breast cancer classifications from 105 breast cancer patients.
    
    Variables: **Complete TCGA ID',
 'Gender',
 'Age at Initial Pathologic Diagnosis',
 'ER Status',
 'PR Status',
 'HER2 Final Status',
 'Tumor',
 'Tumor--T1 Coded',
 'Node',
 'Node-Coded',
 'Metastasis',
 'Metastasis-Coded',
 'AJCC Stage',
 'Converted Stage',
 'Survival Data Form',
 'Vital Status',
 'Days to Date of Last Contact',
 'Days to date of Death',
 'OS event',
 'OS Time',
 'PAM50 mRNA',
 'SigClust Unsupervised mRNA',
 'SigClust Intrinsic mRNA',
 'miRNA Clusters',
 'methylation Clusters',
 'RPPA Clusters',
 'CN Clusters',
 'Integrated Clusters (with PAM50)',
 'Integrated Clusters (no exp)',
 'Integrated Clusters (unsup exp)'**


3. BCGENES.csv 

    This dataset was generated using subset reduction technique to trim down noisy gene variables and dylpr to inner_\join variables from the 77_cancer_proteomes_CPTAC_itraq.csv and clinical_data_breast_cancer.csv
    
    Variables: **'PAM50 mRNA',
 'myoferlin isoform a',
 'heat shock protein HSP 90-beta isoform a',
 'keratin, type II cytoskeletal 72 isoform 1',
 'dedicator of cytokinesis protein 1',
 'keratin, type I cytoskeletal 23',
 '1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta-3 isoform 1',
 'TBC1 domain family member 1 isoform 2',
 'kinesin-like protein KIF21A isoform 1',
 'ubiquitin carboxyl-terminal hydrolase 4 isoform a',
 'PREDICTED: myomegalin-like',
 'KN motif and ankyrin repeat domain-containing protein 1 isoform a',
 'spermatogenesis-associated protein 5',
 'TBC1 domain family member 9',
 'receptor tyrosine-protein kinase erbB-2 isoform a precursor',
 'epidermal growth factor receptor isoform a precursor',
 'UPF0505 protein C16orf62',
 'MICAL-like protein 1',
 'UTP--glucose-1-phosphate uridylyltransferase isoform a',
 'probable ATP-dependent RNA helicase DDX6',
 'L-lactate dehydrogenase B chain',
 'myelin expression factor 2',
 'histone-lysine N-methyltransferase NSD3 isoform long',
 'histone-lysine N-methyltransferase NSD3 isoform short',
 'signal transducer and activator of transcription 6 isoform 1',
 'WD repeat-containing protein 91',
 'UPF0553 protein C9orf64',
 'ran-binding protein 9',
 'arfaptin-1 isoform 2',
 'tonsoku-like protein',
 'protein LSM14 homolog B',
 'oxysterol-binding protein-related protein 2 isoform 2',
 'PDZ domain-containing protein GIPC2',
 'calpain small subunit 2',
 'condensin-2 complex subunit G2',
 'telomere length regulation protein TEL2 homolog',
 'guanine nucleotide-binding protein G(olf) subunit alpha isoform 1',
 'squalene synthase',
 'cathepsin B preproprotein',
 "5'-nucleotidase domain-containing protein 2 isoform 1",
 'COBW domain-containing protein 1 isoform 2',
 'transcription elongation factor A protein-like 5',
 'DNA repair protein XRCC4 isoform 1',
 'transmembrane protein 132A isoform a precursor',
 'alpha-methylacyl-CoA racemase isoform 3',
 'trans-acting T-cell-specific transcription factor GATA-3 isoform 1',
 'uncharacterized protein C7orf43',
 'molybdopterin synthase catalytic subunit large subunit MOCS2B',
 'Golli-MBP isoform 1',
 'hepatocyte nuclear factor 3-gamma',
 '39S ribosomal protein L40, mitochondrial',
 'migration and invasion enhancer 1',
 'ubiquitin-conjugating enzyme E2 E3',
 'protein S100-A13',
 'transcription initiation factor TFIID subunit 8',
 'phosphoribosyltransferase domain-containing protein 1',
 'polyadenylate-binding protein-interacting protein 2'].**

### Manipulation of Breast Cancer Proteomes and Clinical Data Breast Cancer datasets in Rstudio

The data used in this portion of the project was gathered from kaggle. The original dataset includes 12553 unique genes from a total of 80 breast cancer patients and 3 healthy individuals. Further examination has identified the variable ‘Complete TCGA ID’ to be in both the 77_cancer_proteomes_CPTAC_itraq.csv dataset and the clinical_data_breast_cancer.csv dataset. The Complete TCGA ID refers to a breast cancer patient, some patients can be found in both datasets. This is vital to our classification problem, since there are clinical records and proteome data for each patient. 

There are two substantial issues with the Complete TCGA ID variable. The first issue being the Complete TCGA ID does not entirely match up in the datasets(AO-A12D.01TCGA
 = TCGA-AO-A12D); The second issue is the Complete TCGA ID in 77_cancer_proteomes_CPTAC_itraq.csv is treated as variables (recorded as columns) whereas the clinical_data_breast_cancer.csv treats each ID as a record (recorded as a row). To solve the first problem, we use the package stringr to manipulate the Complete TCGA ID so that they follow a similar syntax in both datasets. To solve the second problem, we can transpose the dataset in excel, reassigning Complete TCGA ID as a row and gene expressions as columns. These transformations need to occur before we can combine datasets. 

The only variable needed from clinical_data_breast_cancer.csv is 'PAM50 mRNA', which is the recorded molecular subtype. Using the package dpylr, we can select the variables: 'PAM50 mRNA' and ‘Complete TCGA ID’ from clinical_data_breast_cancer.csv and inner join both datasets by a unique Complete TCGA ID. The result of the dataset manipulation is that we now have a single data set that encompasses 12553 unique genes and the molecular tumor type from 80 patients. 

** Since we transpose the 77_cancer_proteomes_CPTAC_itraq.csv, we cannot use the original dataset, or else inner joining the two datasets will fail. I have uploaded transposed version of the 77_cancer_proteomes_CPTAC_itraq.csv dataset which will be used.**

### Variable Reduction of the 77_cancer_proteomes_CPTAC_itraq.csv dataset using Lasso 
For classification purposes, it is not ideal to include all 12553 unique genes in a classification model, as overfitting can yield misleading results.   

It is common that genetic datasets exhibit cases where n, the number of observations is larger than the p, the number or predictors. When p >n, there is no longer a unique least squares coefficient estimate. Often when p>n, high
variance and overfitting are a major concern in this setting. Thus,
simple, highly regularized approaches often become the methods of choice. To address this issue, there are many techniques (Subset selection), (Lasso and Ridge) and (Dimension reduction) to exclude irrelevant variables from a regression or a dataset. 

Using a Shrinkage Method known as Lasso in Rstudio, we fit a model containing all 12553 predictors
that constrains or regularizes the coefficient estimates, or equivalently, that
shrinks the coefficient estimates towards zero. Variables with a coefficient estimate of zero are left out of the final model. **The results of the lasso method on the 77_cancer_proteomes_CPTAC_itraq.csv indicates 56 relevant variables that will be used in predicting tumor type and for data visualization.**

The significant 56 genes from a sample of breast cancer patients along with the associated tumor types are then queried out in RStudio into dataset BCGENES.csv. **BCGENES.csv is the main dataset which will be used for classification .** A variable importance plot from Rstudio of the cancer data indicates the top 5 most influential genes we will use for data visualization purposes.

# Part 3: Loading Packages and Reading in datasets in Rstudio
library(stringr) #assist with text manipulation
library(dplyr) # data manipulation
library(readr) # data input
library(caret) #select tuning parameters
library(MASS) # contains the data
library(nnet) # used for Multinomial Classification 
library(readr) #assist with text manipulation
library(kernlab) #assist with SVM feature selection
library(class) # used for an object-oriented style of programmin
library(KernelKnn) # used for K- Nearest-Neighbors method
library(nnet) # Used for Neural Net
library(e1071) 
library(gbm)
library(xgboost) # Used for xgbTree
SSC <-read_csv("../input/transposed/77_cancer_proteomes_CPTAC_itraq.csv")
dim(SSC) #86 records, 12553 predictors
clin <-read_csv("../input/editpam50/clinical_data_breast_cancer.csv")
dim(clin) #105 Records, 30 clinical predictors/measurements
# Part 4: Using Stringr to Manipulate Complete TCGA ID

 We can combine datasets if the rows share a unique id/reference number; both datasets share a unique id value (Complete TCGA ID). We will use the Complete TCGA ID variable to combine these datasets together. However, the Complete TCGA ID does not entirely match up in the datasets, where AO-A12D.01TCGA
 = TCGA-AO-A12D.  To combine the clinical and proteomes datasets we use the package stringr to manipulate the Complete TCGA ID so that they follow a similar syntax in both datasets. 
 
 The code below manipulates the Complete TCGA ID number in the Proteomes dataset, queries the Complete TCGA ID number and Molecular tumor type from the Clinical dataset and combines the Proteomes and Clinical dataset as **MASTER**.
p1<-str_replace(SSC$`Complete TCGA ID`,"TCGA","")
p2<-str_replace(p1,".\\d+$","")
p3<-str_replace(p2,"^","TCGA-")

for (i in seq(1:83)){
  SSC$`Complete TCGA ID`[i]<-p3[i]
}

colnames(clin)
clinp<-clin[,c(1,21)]

MASTER<- inner_join(SSC,clinp)
# Part 5: Using Only Variables/Predictors With At Least 99% Of The Data

The data frame **MASTER** contains all variables needed for tumor classification. There is an issue of some gene expressions having missing data. To account for missing data, we identify  gene expressions with at least 99% of data and store them in the **CDF** data frame. Any gene expressions with less than 99% of the data does not meet our criteria and is excluded from the **CDF** data frame. 
#use only variables with 99% of the data
pcent<-sapply(MASTER, function(x) sum(!is.na(x))/length(x))
              
clean <- MASTER[1]
for (i in seq(2, 12554))
{
  if(pcent[i]>0.99){
    clean<-cbind(clean,MASTER[i])
  }
}

# Remove any NA's in the data frame,change to 0
clean[is.na(clean)] <- 0
CDF<-clean

#Remove the Complete TCGA ID from the CDF data frame.
# No longer needer for molecular classification
CDF<-CDF[-1]
attach(CDF)
# Part 6: Using Lasso For Variable/SubSet Selection.

It is common that genetic datasets exhibit cases where n, the number of observations is larger than the p, the number or predictors. When p >n, there is no longer a unique least squares coefficient estimate. Often when p>n, high variance and overfitting are a major concern in this setting. Thus, simple, highly regularized approaches often become the methods of choice. To address this issue, there are many techniques (Subset selection), (Lasso and Ridge) and (Dimension reduction) to exclude irrelevant variables from a regression or a dataset.

Using a Shrinkage Method known as Lasso in Rstudio, we fit a model containing all 12553 predictors that constrains or regularizes the coefficient estimates, or equivalently, that **shrinks the coefficient estimates towards zero. Variables with a coefficient estimate of zero are left out of the final model.** 

The results of the Lasso data reduction technique are too large to be view in the Rstudio console. Therefore the results exported into a txt file on your local machine. 
## Randomly dividing our dataset, which encompasses 4 major molecular tumor types, into datasets for testing and training out model. 
library(glmnet)
CDF$`PAM50 mRNA`<- as.factor(CDF$`PAM50 mRNA`)
x<-model.matrix(CDF$`PAM50 mRNA`~., CDF)
train<-sample(1:nrow(x), nrow(x)/2)
test<-(-train)
y<-CDF$`PAM50 mRNA`
y.test<-y[test]
set.seed(1) 


#Setting up parameters 
grid <- 10^seq(10,-2, length =100)
CDF$`PAM50 mRNA`<- as.factor(CDF$`PAM50 mRNA`)
set.seed (1)
cv.out<-cv.glmnet(x[train,],y[train],alpha=1,lambda =grid,family="multinomial",type.multinomial="grouped")
plot(cv.out)
bestlam<-cv.out$lambda.min

#  The results of the Lasso data reduction technique are exported onto your local machine for optimal viewing.
out<-glmnet(x,y,alpha =1, lambda=grid,family="multinomial",type.multinomial="grouped")
lasso.coef<-predict(out,type ="coefficients",s=bestlam,family="multinomial",type.multinomial="grouped")
sink(file="lasso.txt")
options("max.print"=8020)
lasso.coef
sink(NULL)
# Part 7: Retrieving  Indexes of Relevant Variables From Lasso 
Upon reviewing the lasso.txt file, there are 58 genetic predictors with non-zero coefficients. These are the predictors that are used to classify molecular tumor type, the following code identity’s the location of each predictor within the dataset and retrieves its column number. 

The column number allows one to directly select a predictor by its column number rather than having to select by name.  **Inputting 1017 is simpler and cleaner than typing out `1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta-3 isoform 1`.** 
#Reading in and printing the column numbers of all genes for final dataset indexing
sink(file="colindex.txt")
options("max.print"=8100)
colnames(CDF)
sink(NULL)

# List of the 56 genetic predictors and the assoicated column number

`myoferlin isoform a` 142
`heat shock protein HSP 90-beta isoform a` 258
`keratin, type II cytoskeletal 72 isoform 1` 283
`cingulin` 342
`dedicator of cytokinesis protein 1` 537
`keratin, type I cytoskeletal 23`774
`1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta-3 isoform 1` 1017
`TBC1 domain family member 1 isoform 2` 1069
`kinesin-like protein KIF21A isoform 1` 1107
`ubiquitin carboxyl-terminal hydrolase 4 isoform a` 1125
`PREDICTED: myomegalin-like` 1164
`KN motif and ankyrin repeat domain-containing protein 1 isoform a` 1280
`spermatogenesis-associated protein 5` 1281
`TBC1 domain family member 9` 1352
`receptor tyrosine-protein kinase erbB-2 isoform a precursor` 1376
`epidermal growth factor receptor isoform a precursor` 1377
`UPF0505 protein C16orf62` 1572
`MICAL-like protein 1` 1643
`UTP--glucose-1-phosphate uridylyltransferase isoform a` 1717
`probable ATP-dependent RNA helicase DDX6` 1848
`L-lactate dehydrogenase B chain` 1916
`myelin expression factor 2` 2037
`histone-lysine N-methyltransferase NSD3 isoform long` 2065
`histone-lysine N-methyltransferase NSD3 isoform short` 2067
`signal transducer and activator of transcription 6 isoform 1` 2540
`WD repeat-containing protein 91` 2726
`UPF0553 protein C9orf64` 3201
`ran-binding protein 9` 3226
`arfaptin-1 isoform 2` 3438
`tonsoku-like protein` 3567
`protein LSM14 homolog B` 3606
`oxysterol-binding protein-related protein 2 isoform 2` 3629
`PDZ domain-containing protein GIPC2` 3807
`calpain small subunit 2` 4075
`condensin-2 complex subunit G2` 4083
`telomere length regulation protein TEL2 homolog` 4088
`guanine nucleotide-binding protein G(olf) subunit alpha isoform 1` 4116
`squalene synthase` 4613
`cathepsin B preproprotein` 4816
`5'-nucleotidase domain-containing protein 2 isoform 1` 4894
`COBW domain-containing protein 1 isoform 2` 4914
`transcription elongation factor A protein-like 5`5218
`DNA repair protein XRCC4 isoform 1` 5542
`transmembrane protein 132A isoform a precursor` 5574
`transcription factor AP-2-beta` 5628
`alpha-methylacyl-CoA racemase isoform 3` 5693
`trans-acting T-cell-specific transcription factor GATA-3 isoform 1` 6111
`uncharacterized protein C7orf43` 6179
`molybdopterin synthase catalytic subunit large subunit MOCS2B` 6533
`Golli-MBP isoform 1` 6581
`hepatocyte nuclear factor 3-gamma` 6769
`39S ribosomal protein L40, mitochondrial`7015
`migration and invasion enhancer 1`7319
`ubiquitin-conjugating enzyme E2 E3` 7331
`protein S100-A13`7357
`transcription initiation factor TFIID subunit 8`7425
`phosphoribosyltransferase domain-containing protein 1`7512
`polyadenylate-binding protein-interacting protein 2`7923
# Part 8: Selecting Predictors By Column Number 

Referring to **Part 5**, we identified gene expressions with at least 99% of data and stored them in the **CDF** data frame. Therefore, the code below selects the predictors that have non-zero coefficients, which were  identified in **part 6**,from the **CDF** data frame and stores these predictors along with the variable for molecular tumor type (`PAM50 mRNA`) into a new data frame **testdf**. 
testdf<-CDF[,c(8018,142,258,283,537,774,1017,1069,1107,1125,1164,1280,1281,1352,
               1376,1377,1572,1643,1717,1848,1916,2037,2065,2067,2540,2726,3201,
               3226,3438,3567,3606,3629,3807,4075,4083,4088,4116,4613,4816,4894,
               4914,5218,5542,5574,5693,6111,6179,6533,6581,6769,7015,7319,
               7331,7357,7425,7512,7923)]
# Part 9: Classifying Molecular Tumor Type Using Ensemble, Multinomial, K Nearest Neighbors, Support Vector Machine, and Other Machine Learning Methods 
## Molecular Tumor Type Using Ensemble-Boosting Regression Tree 

Gradient boosting is a machine learning technique for regression and classification problems, the basis involves 3 broad elements. 

1. A loss function to be optimized.
2. A weak learner to make predictions.
3. An additive model to add weak learners to minimize the loss function.

Gradient boosting produces a prediction model in the form of an ensemble of weak prediction models. 

The code below uses the package **gbm**.  The **first figure** shows the relative information value of all variables in numeric form. Indicating that `hepatocyte nuclear factor 3-gamma` and `signal transducer and activator of transcription 6 isoform 1` are heavily influential in determining molecular tumor type. The **second figure** shows the relative information value on a horizontal bar plot. The **third figure** shows the lowest number of iterations with the lowest multinomial Deviance. Essentially is quantifies the accuracy of a classifier by penalizing false classifications, the goal is to minimize the multinomial deviance and increase accuracy.   

The second block of code is where the  Ensemble-Boosting method is applied to our data. We run the model 20 times and compute the average the accuracy, this is the best estimate since the range is extensive. 
testdf$`PAM50 mRNA`<- as.factor(testdf$`PAM50 mRNA`)

gbm.model<-gbm(`PAM50 mRNA`~., data=testdf, shrinkage=0.01, distribution = 'multinomial',
               cv.folds=5, n.trees=5000, verbose=F)
summary(gbm.model)
best.iter<-gbm.perf(gbm.model, method="cv")
fitControl <- trainControl(method="cv", number=10, returnResamp = "all",classProbs = TRUE, summaryFunction = multiClassSummary)
AccuraciesGLM<- c(0,0,0)
for(i in seq(20)){
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  trainDF$`PAM50 mRNA`<-as.factor(trainDF$`PAM50 mRNA`)
  testDF$`PAM50 mRNA`<-as.factor(testDF$`PAM50 mRNA`)
  gbm.model<-gbm(`PAM50 mRNA`~., data=testdf, shrinkage=0.01, distribution = 'multinomial',
                 cv.folds=5, n.trees=5000, verbose=F)
  best.iter<-gbm.perf(gbm.model, method="cv")
  
  obj<-train(`PAM50 mRNA`~., data=trainDF, method="gbm",distribution="multinomial",
             trControl=fitControl, verbose=F,metric = "ROC",
             tuneGrid=data.frame(.n.trees=best.iter, .shrinkage=0.01, .interaction.depth=4, .n.minobsinnode=1))
  
  AccuraciesGLM[i] <- confusionMatrix(predict(obj,newdata=testDF[-1]),testDF$`PAM50 mRNA`)$overall["Accuracy"]
}
plot(density(AccuraciesGLM))
summary(AccuraciesGLM)
## Molecular Tumor Type Using K- Nearest-Neighbors 

K- Nearest-Neighbors is a non-parametric used for classification/pattern recognition. Before using KNN, all variables are to be normalized and scaled appropriately. To use KNN; one must select the number cluster neighbors (k), the algorithm than uses a weighted average of k nearest neighbors and computes an Euclidean distance. Values near to one another contribute more to the average than distant values. KNN classifies based on how similar our testing data is to the training set. One of the disadvantages of KNN is that it is relies heavily on human input, we must choose the number on neighbors for classification purposes. Since KNN is a non-parametric method, group distance is unimportant since we are not concerned about the distribution of the data, also we can used mixed data, however KNN does not provide a model. 

library(class)
library(KernelKnn)
AccuraciesKNN<-c(0,0,0)
#choose which measurements to use in classification
tf<-testdf
x<-tf[2:57]
#choose which group labels to use in classification
y<-factor(testdf$`PAM50 mRNA`)

for (i in seq(300)){
  kcres3 <-knn.cv(x, y, k = 3, prob= TRUE)
  AccuraciesKNN[i] <-confusionMatrix(kcres3,testdf$`PAM50 mRNA`)$overall["Accuracy"]
}
summary(AccuraciesKNN)
plot(density(AccuraciesKNN))
## Molecular Tumor Type Using Support Vector Machines

Support Vector Machines are generalized extension of a maximal margin classifier, SVM are intended for the binary classification setting when there are two classes. However, this designed intention does not disqualify from using the SVM method with cases of more than two classes. SVM determines the best line separator by identifying closest points in Convex hull, a hyperplane bisects the closest point to the convex hull. The support vector classifies a test observation depending on which side of a plane it lies; this is based on boundaries-support vectors. SVM method allows some observations to be on the incorrect side of the margin and in some cases the incorrect side of the hyperplane in the interest of performing better in classifying the remaining observations further away from the hydroplane. This is known as a soft margin classifier; training observations can violate this area. Advantages of using a SVM  model are; can be adapted to work well with nonlinear boundaries, uses kernels, less overfitting of data, performs well with clear margin of separation among data.
library(kernlab)
AccuraciesSVM <- c(0,0,0)
for(i in seq(1000)){
  
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  
  train.ksvm<-ksvm(`PAM50 mRNA` ~.,
                   scale = TRUE,
                   data=trainDF,
                   kernel="rbfdot",
                   prob.model=TRUE)
  
  AccuraciesSVM[i] <- confusionMatrix(testDF$`PAM50 mRNA`, predict(train.ksvm, testDF))$overall["Accuracy"]
}
summary(AccuraciesSVM)
plot(density(AccuraciesSVM))
## Molecular Tumor Type Using Naïve Bayes Methods

The Naïve Bayes Algorithm is a classifier based on applying Bayes theorem with independent assumptions between features. Meaning that all features in the data set are equally important and independent of one another. Bayesian probability Is rooted in the theory that the likelihood of an event should be based on the evidence across multiple trials. Naïve Bayes uses probabilities to classify groups based on prior probability. One advantage is that Naïve Bayes works with mixed data: nominal, continuous and ordinal variables. Naïve Bayes is fast and effective, handles missing and noisy data well, and requires few records for training and can also work well with large records.  Disadvantages of Naïve Bayes is that it assumes that all the data predictors are independent when in data is far from this faulty assumption. Also, estimated probabilities are less reliable than predicted classes. 
AccuraciesNaive <- c(0.00)
for (i in seq(100)) 
{
 
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  nb1 <- train(`PAM50 mRNA`~., data = trainDF, method = "nb",
               trControl = trainControl(method = "cv"),
               tuneGrid = data.frame(usekernel = TRUE, fL = 0.5, adjust = 5))
  bps <- predict(nb1, newdata=testDF)
  AccuraciesNaive[i] <- confusionMatrix(testDF$`PAM50 mRNA`,bps)$overall["Accuracy"]
}
plot(density(AccuraciesNaive))
summary(AccuraciesNaive)
## Molecular Tumor Type Using Multinomial Method

The Multinomial method is a classifier that is used to predict the probabilities of the different possible outcomes of a categorically dependent variable. One assumption is that collinearity among independent variables are relatively low.
AccuraciesMultinomial<- c(0,0,0)
for(i in seq(1000)){
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  
  net<-multinom(`PAM50 mRNA`~.,
                data=trainDF,trace=FALSE)
  AccuraciesMultinomial[i] <- confusionMatrix(predict(net, newdata=testDF, "class"),testDF$`PAM50 mRNA`)$overall["Accuracy"]
}
plot(density(AccuraciesMultinomial))
summary(AccuraciesMultinomial)
## Molecular Tumor Type Using eXtreme Gradient Boosting

XGBoost (eXtreme Gradient Boosting) is a machine learning classifier/predictor, which produces a model in a form of an esemble of weak prediction models. XGBoost helps to reduce overfitting. 
xgbGrid <- expand.grid(nrounds = c(1, 10),
                       max_depth = c(1, 4),
                       eta = c(.1, .4),
                       gamma = 0,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = c(.8, 1))

cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                       classProbs = TRUE)

AccuraciesXGBTREE <- c(0.00)
for (i in seq(100)) 
{
  
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  trainDF$`PAM50 mRNA`<- as.factor(trainDF$`PAM50 mRNA`)
  
  xgb <- train(`PAM50 mRNA` ~., data=trainDF, 
                               method = "xgbTree", 
                               trControl = cctrl1,
                               preProc = c("center", "scale"),
                               tuneGrid = xgbGrid)
  
  bpred <- predict(xgb, newdata=testDF)
  AccuraciesXGBTREE[i] <- confusionMatrix(testDF$`PAM50 mRNA`,bpred)$overall["Accuracy"]
}

plot(density(AccuraciesXGBTREE))
summary(AccuraciesXGBTREE)
## Molecular Tumor Type Using Neural Network

Inspired by biological neural networks, the Neural Network method is a supervised machine learning algorithm which consists of units arranged in layers which coverts an input vector (independent variable) into a prediction/classification.  "The algorithm learns a function by training on a dataset without prior knowledge about the dataset." 
cctrlR <- trainControl(method = "cv", number = 3, returnResamp = "all", search = "random")

AccuraciesNNET <- c(0.00)
for (i in seq(100)) 
{
  
  train<- createDataPartition(testdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-testdf[train,]
  testDF<-testdf[-train,]
  trainDF$`PAM50 mRNA`<- as.factor(trainDF$`PAM50 mRNA`)
  
  nn <- train(`PAM50 mRNA` ~., data=trainDF, 
               method = "nnet", 
               trControl = cctrl1,
               preProc = c("center", "scale"),
               trace = FALSE)
  
  nnpred <- predict(nn, newdata=testDF)
  AccuraciesNNET[i] <- confusionMatrix(testDF$`PAM50 mRNA`,nnpred)$overall["Accuracy"]
}

plot(density(AccuraciesNNET))
summary(AccuraciesNNET)
## Molecular Tumor Type Using eXtreme Gradient Boosting With Random Variables

One way to test and visualize how accurately the models perform is to generate a separate model in which the independent variables are randomly selected. Previously, independent variables with a coefficient estimate greater than zero were selected using the LASSO method- this method performs variable selection to increase accuracy of a prediction. 

The code below is used to randomly generate 57 variables for our random model.  The values are the column id's of variables and is listed below.

sample(1:8018,57,replace=F)

8018,3870,6845,341,5595,1149,4346,4628,6191,5801,6301,258,
7377,3483,692,6462,335,1217,2216,5857,7707,6501,4610,2605,51,
3755,7888,2322,716,4009,2581,2623,2660,6622,3263,7045,6532,1207,4558, 6232,6943,7340,6546,669,4254,579,6689,6502,4095,5355,6675,842,4693,3991,4890,6887,4782
rdtestdf <-CDF[,c(8018,3870,6845,341,5595,1149,4346,4628,6191,5801,6301,258,
                  7377,3483,692,6462,335,1217,2216,5857,7707,6501,4610,2605,51,
                  3755,7888,2322,716,4009,2581,2623,2660,6622,3263,7045,6532,1207,4558, 
                  6232,6943,7340,6546,669,4254,579,6689,6502,4095,5355,6675,842,4693,3991,
                  4890,6887,4782
                  )]

xgbGrid <- expand.grid(nrounds = c(1, 10),
                       max_depth = c(1, 4),
                       eta = c(.1, .4),
                       gamma = 0,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = c(.8, 1))

cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
                       classProbs = TRUE)

AccuraciesXGBTREERD <- c(0.00)
for (i in seq(100)) 
{
  
  train<- createDataPartition(rdtestdf$`PAM50 mRNA`, p = .70, list = FALSE)
  trainDF<-rdtestdf[train,]
  testDF<-rdtestdf[-train,]
  trainDF$`PAM50 mRNA`<- as.factor(trainDF$`PAM50 mRNA`)
  
  xgb <- train(`PAM50 mRNA` ~., data=trainDF, 
               method = "xgbTree", 
               trControl = cctrl1,
               preProc = c("center", "scale"),
               tuneGrid = xgbGrid)
  
  bpred <- predict(xgb, newdata=testDF)
  AccuraciesXGBTREERD[i] <- confusionMatrix(testDF$`PAM50 mRNA`,bpred)$overall["Accuracy"]
}

plot(density(AccuraciesXGBTREERD))
summary(AccuraciesXGBTREERD)
## Part 10: Conclusions 

Below are the mean results from the machine learning methods,  the models are ran mutlptile times so that one can compute the average the accuracy. 

### Support Vector Machines(SVM): 97%
### Neural Network:  96%
### Multinomial Method: 90%
### K- Nearest-Neighbors (KNN): 89%
### Gradient Boosting Machine (GBM): 86%
### Naïve Bayes: 82%
### eXtreme Gradient Boosting: 77%
### eXtreme Gradient Boosting with Random Variables: 55%

The highest mean classification accuracy resulted from the Support Vector Machines(SVM) and Neural Network model. Models that apply selected predictors from variable selection outperformed the eXtreme Gradient Boosting model with Random Variables in terms of classification accuracy. 

Tumors will often vary in location, size, shape, and grade(severity). These characteristics, along with hormone receptor status and HER2 status affect prognosis and overall survival rate, the percentage of patients who are still alive for certain period after they were diagnosed with or started treatment for a disease, such as cancer. Currently, research has identified 4 major molecular subtypes within breast cancer tumors, these subtypes are based on the genes that cancer cell express: Luminal A, Luminal B, HER2 Enriched, and Basal (triple negative). Identifying and studying these subtypes has potential in planning more effective treatment optand developing new therapies. 

