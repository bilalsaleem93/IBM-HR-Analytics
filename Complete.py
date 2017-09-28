import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pandas.tools.plotting import scatter_matrix
#import seaborn as sns
from sklearn import datasets 
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import  cross_val_score
from sklearn import grid_search


# Extract Data from .csv File and assign appropriate headers

def extract(filename):
   
    data = pd.read_csv(filename,sep=",",decimal=".",header=None,names=['Satis_level','Last_eval','No_proj','Avg_monthly-hr','Time_company','Work_accident','Left','Prom_5yrs','Dept','Salary'])
    data = data.drop(0)
    #print len(data)
    #print 'Data Types of each column are '+str(data.dtypes)
    return data


# Make Sure that all series are having correct Data Type

def ensure_data_types(data):
    
    S_L = data["Satis_level"].str.replace(' ', '')
    data["Satis_level"]= pd.to_numeric(S_L,errors='coerce')
    
    L_E = data["Last_eval"].str.replace(' ', '')
    data["Last_eval"]= pd.to_numeric(L_E,errors='coerce')
    
    N_P = data["No_proj"].str.replace(' ', '')
    data["No_proj"]= pd.to_numeric(N_P,errors='coerce')
    
    AM_H = data["Avg_monthly-hr"].str.replace(' ', '')
    data["Avg_monthly-hr"]= pd.to_numeric(AM_H,errors='coerce')
    
    T_C = data["Time_company"].str.replace(' ', '')
    data["Time_company"]= pd.to_numeric(T_C,errors='coerce')
    
    W_A = data["Work_accident"].str.replace(' ', '')
    data["Work_accident"]= pd.to_numeric(W_A,errors='coerce')
    
    Label = data["Left"].str.replace(' ', '')
    data["Left"]= pd.to_numeric(Label,errors='coerce')
    
    Promotion = data["Prom_5yrs"].str.replace(' ', '')
    data["Prom_5yrs"]= pd.to_numeric(Promotion,errors='coerce')
    
    #print 'Data Types of each column are '+str(data.dtypes)

# Make Sure that there are no Typos in any series
    
def Typos(data):
    
    #print sum(data['Dept'].value_counts())
    #print sum(data['Salary'].value_counts())
    #print (data['Satis_level'].value_counts())
    #print data['Last_eval'].value_counts()
    #print (data['No_proj'].value_counts())
    #print (data['Avg_monthly-hr'].value_counts())
    #print (data['Work_accident'].value_counts())
    #print (data['Left'].value_counts())
    #print (data['Prom_5yrs'].value_counts())
    #print  (data['Time_company'].value_counts())
    print 'No Typos Found'
    
# Remove White Spaces from series having String dtype    
    
def White_spaces(data):
    
    data["Dept"] = data["Dept"].str.strip()
    data["Dept"] = data ["Dept"].str.lower()
    
    data["Salary"] = data["Salary"].str.strip()
    data["Salary"] = data["Salary"].str.lower()
    
# Check all numeric series for Insane Values    
    
def Sanity_check(data):
    
    if (data["Satis_level"].any() > 1 | data["Satis_level"].any() < 0 ):
        print 'Insane Value in Satis_level'
        
    if (data["Last_eval"].any() > 1 | data["Last_eval"].any() < 0 ):
        print 'Insane Value in Last_eval'
        
    if (data["No_proj"].any() < 0 ):
        print 'Insane Value in No_proj'

    if (data["Work_accident"].any() != 1 | data["Work_accident"].any() != 0 ):
        print 'Insane Value in Work_accident'
        
    if (data["Prom_5yrs"].any() != 1 | data["Prom_5yrs"].any() != 0 ):
        print 'Insane Value in Prom_5yrs'

    if (data["Left"].any() != 1 | data["Left"].any() != 0 ):
        print 'Insane Value in Left'

    if (data["Time_company"].any() < 0 | data["Time_company"].any() > 40 ):
        print 'Insane Value in Time_company'
        
    if (data["Avg_monthly-hr"].any() < 0 | data["Time_company"].any() > 380 ):
        print 'Insane Value in Avg_monthly-hr'
        

def Relationship_attributes(data):
        #Boxplot Satisfaction Level vs left
    data.dropna().boxplot(column='Satis_level',by='Left')
    plt.title('Satisfaction Level vs Left')
    plt.ylabel('Satisfaction Level')
    plt.xlabel('Left')
    plt.show()

    #Boxplot Last Evaluation vs left
    data.dropna().boxplot(column='Last_eval',by='Left')
    plt.title('Last Evaluation vs Left')
    plt.ylabel('Last Evaluation')
    plt.xlabel('Left')
    plt.show()

    #Box plot Number of Projects vs left
    data.dropna().boxplot(column='No_proj',by='Left')
    plt.title('Number of Projects vs Left')
    plt.ylabel('Number of Projects')
    plt.xlabel('Left')
    plt.show()

    #Box plot Average Monthly Hours vs left
    data.dropna().boxplot(column='Avg_monthly-hr',by='Left')
    plt.title('Average Monthly Hours vs Left')
    plt.ylabel('Average Monthly Hours')
    plt.xlabel('Left')
    plt.show()

    #Box plot Time spend in the Company vs left
    data.dropna().boxplot(column='Time_company',by='Left')
    plt.title('Time spend in the Company vs Left')
    plt.ylabel('Time spend in the Company')
    plt.xlabel('Left')
    plt.show()

    #Left filtered
    left0 = data[data['Left']==0]
    left1 = data[data['Left']==1]

    #Histogram Satisfaction Level vs Left
    left0.Satis_level.plot(kind='hist',bins=100)
    left1.Satis_level.plot(kind='hist',bins=100)
    plt.legend(labels=('Present','Left'))
    plt.title('Satisfaction Level - Left')
    plt.xlabel('Satisfaction Level')
    plt.show()

    #Histogram Last Evaluation vs Left
    left0.Last_eval.plot(kind='hist',bins=100)#,legend='0')
    left1.Last_eval.plot(kind='hist',bins=100)#,legend='1')
    plt.legend(labels=('Present','Left'))
    plt.title('Last Evaluation - Left')
    plt.xlabel('Last Evaluation')
    plt.show()

    #Histogram Time spent in the Company vs Left
    left0.Time_company.plot(kind='hist',bins=10)#,legend='0')
    left1.Time_company.plot(kind='hist',bins=10)#,legend='1')
    plt.legend(labels=('Present','Left'))
    plt.title('Time spent in the Company - Left')
    plt.xlabel('Time spent in the Company')
    plt.show()

    #Histogram Work Accident vs Left
    left0.Work_accident.plot(kind='hist',bins=2)#,legend='0')
    left1.Work_accident.plot(kind='hist',bins=2)#,legend='1')
    plt.legend(labels=('Present','Left'))
    plt.title('Work Accident - Left')
    plt.xlabel('Work Accident')
    plt.show()

    #Histogram Promotions vs Left
    left0.Prom_5yrs.plot(kind='hist',bins=2)#,legend='0')
    left1.Prom_5yrs.plot(kind='hist',bins=2)#,legend='1')
    plt.legend(labels=('Present','Left'))
    plt.title('Promotions- Left')
    plt.xlabel('Promotions')
    plt.show()


    #Salary Filtered
    low=data[data['Salary']=='low']
    medium=data[data['Salary']=='medium']
    high=data[data['Salary']=='high']

    #Density Satisfaction Level vs Salary
    low.Satis_level.plot(kind='density',legend='Low')
    medium.Satis_level.plot(kind='density',legend='Medium')
    high.Satis_level.plot(kind='density',legend='High')
    plt.title('Satisfaction Level by Salary')
    plt.xlabel('Satisfaction Level')
    plt.show()

    #Density Last Evaluation vs Salary
    low.Last_eval.plot(kind='density',legend='Low')
    medium.Last_eval.plot(kind='density',legend='Medium')
    high.Last_eval.plot(kind='density',legend='High')
    plt.title('Last Evaluation by Salary')
    plt.xlabel('Last Evaluation')
    plt.show()

    #Sales vs Left
    #sns.countplot(y='Left',hue='Dept',data=data)
    #plt.show()

    #Salary vs Number of Projects
    #sns.countplot(y='Salary',hue='No_proj',data=data)
    #plt.show()

    #Scatter Matrix
    scatter_matrix(data,alpha=0.2,figsize=(16,16),diagonal='hist')
    plt.show()
    
def single_attributes(data):
    
    left = data["Left"].value_counts()
    l_legends,_,_ = plt.pie(left,shadow=True,colors= ['lightcoral','lightskyblue'],autopct='%.2f')
    plt.legend(l_legends,labels = ('Present','Ex Employees'))
    plt.title(' Present vs Ex Employees')
    plt.axis('equal')
    plt.show()
    
    prom = data["Prom_5yrs"].value_counts()
    prom_legends,_,_ = plt.pie(prom,shadow=True,colors= ['coral','lightgreen'],autopct='%.2f')
    plt.legend(l_legends,labels = ('Not Promoted','Promoted'))
    plt.title(' Promotion in Last 5 Years')
    plt.axis('equal')
    plt.show()
    
    sal = data["Salary"].value_counts()
    sal_legends,_,_ = plt.pie(sal,shadow=True,colors= ['lightgrey','lightgreen','lightblue'],autopct='%.2f')
    plt.legend(sal_legends,labels = ('Low','Medium','High'))
    plt.title(' Salary ')
    plt.axis('equal')
    plt.show()
    
    #sns.countplot(y= 'Dept',hue='Dept',data=data)
    #plt.show()
    
    
    data['Satis_level'].plot(kind='density')
    plt.title('Satisfactation Level of Employees')
    plt.xlabel('Satisfactation')
    plt.show()
    
    data['Last_eval'].plot(kind='density')
    plt.title('Score in Last Evaluation of Employees')
    plt.xlabel('Score')
    plt.show()
    
    data['No_proj'].value_counts().plot(kind='bar',align='center',alpha = 0.5)
    plt.title('Number of Projects assigned to Employee')
    plt.xlabel('Number of Projects')
    plt.show()
    
    data.dropna().boxplot(column='Avg_monthly-hr')
    plt.title('Average Monthly Hours')
    #plt.xlabel('Average Monthly Hours')
    plt.ylabel('Hours')
    plt.show()
    
    data['Time_company'].value_counts().plot(kind='bar',align='center',alpha = 0.5)
    plt.title('Time Spent in Company')
    plt.ylabel('No. of Employees')
    plt.xlabel('Years')
    plt.show()
    
    w_a = data["Work_accident"].value_counts()
    w_legends,_,_ = plt.pie(w_a,shadow=True,colors= ['lightcoral','lightskyblue'],autopct='%.2f')
    plt.legend(w_legends,labels = ('No Accident','Accident'))
    plt.title(' Work Accident at Company ')
    plt.axis('equal')
    plt.show()

# Seperate Series for Label. Later on to be used for Classification
        
def Generate_labels(data):

    labels = pd.Series(data["Left"].copy(), index=data["Left"].index.copy(), name=data["Left"].name)
    data.drop('Left',axis = 1, inplace=True)
    #print labels
    return labels
    
    #if (Labels.all() == data["Left"].all()):
     #   print "yes"
#def masks(A,B,data):
   # mask = data['Dept'] is A.strip()
   # data.loc[mask,B] = 1
    #print data['Dept'].value_counts()

def category(data):
    
    mask = data["Salary"] == "low"
    data.loc[mask,'Salary'] = 0
    mask2 = data["Salary"] == "medium"
    data.loc[mask2,'Salary'] = 1
    mask3 = data["Salary"] == "high"
    data.loc[mask3,'Salary'] = 2
    
    ###   Now Dept  ###
    
    Dept_len = len(data['Dept'])
    data['Account'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Hr'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['It'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Mgmt'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Markt'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Pgm'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Rnd'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Sales'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Support'] = pd.Series(np.zeros(Dept_len), index=data.index)
    data['Tech'] = pd.Series(np.zeros(Dept_len), index=data.index)
    acc = data["Dept"] == "accounting"
    data.loc[acc,'Account'] = 1
    
    hr = data["Dept"] == "hr"
    data.loc[hr,'Hr'] = 1
    
    it = data["Dept"] == "it"
    data.loc[it,'It'] = 1    
    
    mgmt = data["Dept"] == "management"
    data.loc[mgmt,'Mgmt'] = 1    
    
    markt = data["Dept"] == "marketing"
    data.loc[markt,'Markt'] = 1
    
    pgm = data["Dept"] == "project_mng"
    data.loc[pgm,'Pgm'] = 1
    
    rnd = data["Dept"] == "randd"
    data.loc[rnd,'Rnd'] = 1
    
    sales = data["Dept"] == "sales"
    data.loc[sales,'Sales'] = 1
    
    supp = data["Dept"] == "support"
    data.loc[supp,'Support'] = 1
    
    tech = data["Dept"] == "technical"
    data.loc[tech,'Tech'] = 1
    
    data.drop('Dept',axis = 1, inplace=True)

    
def KNN_Classifier(data,y):
    
    n_samples=len(data)
    #print n_samples
    X = np.array(data).reshape((n_samples,-1))
    label = np.array(y)
    #k-fold cross validation
    
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.30,random_state=0)
    k_range = range(1, 10)
    cv_scores = []
    myList = list(range(1,50))
    neighbors = filter(lambda x: x % 2 != 0, myList)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10,scoring='accuracy')
        cv_scores.append(scores.mean())
    
    MSE = [1 - x for x in cv_scores]

    # determining best k
    best_k = neighbors[MSE.index(min(MSE))]
    print "The optimal number of neighbors is %d" % best_k
    
    
    
    
    clf = KNeighborsClassifier(best_k,weights='uniform',metric = 'minkowski',p=2)
    fit = clf.fit(X_train,y_train)
    predicted = fit.predict(X_test)
    print confusion_matrix(y_test,predicted)
    print classification_report(y_test,predicted) 
    
def Decision_Tree(data,y):
    n_samples=len(data)
    X = np.array(data).reshape((n_samples,-1))  
    label = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.4,random_state=0) 
    
    ### For best parameters ###
    parameters = {'max_depth':range(3,15),'min_samples_split':range(10,30)}
    kclf = grid_search.GridSearchCV(DecisionTreeClassifier(), parameters,cv=10)
    kclf.fit(data,y)
    ktree_model = kclf.best_estimator_
    print (kclf.best_params_) 
    
    
    clf = DecisionTreeClassifier(criterion='gini',max_features="auto",max_depth=kclf.best_params_['max_depth'],min_samples_split=kclf.best_params_['min_samples_split'])
    fit = clf.fit(X_train, y_train)
    y_pre = fit.predict(X_test)
    print confusion_matrix(y_test, y_pre) 
    print classification_report(y_test,y_pre) 

def SVM(data,y):
    n_samples=len(data)
    X = np.array(data).reshape((n_samples,-1))  
    label = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.4,random_state=0)
    clf = SVC(C = 2.6,class_weight='balanced')
    fit = clf.fit(X_train,y_train)
    y_pre = fit.predict(X_test)
    print confusion_matrix(y_test, y_pre)
    print classification_report(y_test,y_pre) 
    
    
        
#####    MAIN    #####
if __name__ == "__main__":
    ### Data Exploration Phase ###
    filename = "HR.csv"  
    data = extract(filename)
    ensure_data_types(data)
    Typos(data)
    White_spaces(data)
    Sanity_check(data)
    Relationship_attributes(data)
    single_attributes(data)
    Labels = Generate_labels(data)
    #print data["Salary"].value_counts()
    category(data)
    print 'Results of KNN'
    KNN_Classifier(data,Labels)
    print 'Results of Decision Tree'
    Decision_Tree(data,Labels)
    print 'Results of SVM'
    SVM(data,Labels)
    #print data["Salary"].value_counts()
  
    ### Data Cleansing Done ###  
    
    ### Now Data Exploration Phase ###
    


