
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from pygame import mixer # Load the required library


def create_arr(answers,data,rows,columns):
    gnb=GaussianNB()
    
    feature_cols = list(data.columns[1:60])
    target_col = data.columns[60]

    # Separate the data into feature data and target data (X_all and y_all, respectively)
    X_all = data[feature_cols]
    y_all = data[target_col]
    num_all = data.shape[0] 
    num_train = int(num_all*0.77) # about 77% of the data
    num_test = num_all - num_train

    # begin splitting data into various sets for comparision
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=num_test, random_state=5)
    print("Shuffling of data into test and training sets complete!")
    print("Training set: {} samples".format(X_train.shape[0]))
    print("Test set: {} samples".format(X_test.shape[0]))    
    gnb.fit(X_train, y_train)
    results=(y_test==gnb.predict(X_test))
    ytrue=0
    yfalse=0
    for result in results:
        if(result):
            ytrue+=1
            mixer.init()
            mixer.music.load(r".\C.mp3")
            mixer.music.play()
    
        else:
            yfalse+=1
    print("True: ",ytrue)
    print("False: ",yfalse)
    print("Score:",ytrue/num_test)



    
def main():
    #get the file
    loc=r".\num.csv"
    data=pd.read_csv(loc)
    a=data.shape
    columns=a[0]
    rows=a[1]
    create_arr([],data,rows,columns) 
       

main()