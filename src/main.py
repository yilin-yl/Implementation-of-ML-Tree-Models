import src

#import train dataset
ftrn = open(input("Enter train.csv and its directory:"),"r")
data = src.readData(ftrn)
ftrn.close()
for row in data:
    row[-1] = src.binary_y(row[-1])

#import test dataset
ftst = open(input("Enter test.csv and its directory:"),"r")
testdata = src.readData(ftst)
ftst.close()
for row in testdata:
    row[-1] = src.binary_y(row[-1])

#use 5-folds cross validation to choose models. (CART or Random Forest)
print(src.cvTree(data,5)) #print cv result - accuracy of CART model.
print(src.cvRF(data,5))  #print cv result - accuracy of random forest model.

#Final approach: random forest with parameter m=3, n=11.
def main():
    rfmodel = src.rf(data,3,11)
    prediction = src.rfPred(rfmodel,testdata)
    truey = src.colList(testdata,-1)
    print(src.accuracy(truey,prediction))

main()