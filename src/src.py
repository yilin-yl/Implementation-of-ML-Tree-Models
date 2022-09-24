#####################Data Preparation#############################
#read csv file.
def readData(f):
    data = []
    for line in f:
        l_line = line.split(",")
        l_line = [a.strip() for a in l_line]
        data.append(l_line)
    for i in range(1,len(data)):
        data[i] = [eval(x) for x in data[i]]
    data.pop(0)
    return data

#convert y into binary values (0 or 1)
def binary_y(yi):
    if yi > 6:
        return 1
    else:
        return 0

######################Decision Tree(CART)##############################

#get a column of dataset
def colList(dataset,colIndex):
    result = [lst[colIndex] for lst in dataset]
    return result

#calculate Gini index for a given list of y 
def Gini(ylist):
    num0 = 0
    for i in ylist:
        if i == 0:
            num0 += 1
    p0 = num0/len(ylist)
    return 2*p0*(1-p0)

#According to the feature and cutpoint, split the dataset into 2. return [subset1,subset2]
def splitData(dataset,colIndex,cutpoint):
    subset1 = []
    subset2 = []
    for i in range(len(dataset)):
        if dataset[i][colIndex] < cutpoint:
            subset1.append(dataset[i])
        else:
            subset2.append(dataset[i])
    return [subset1,subset2]

#find the best split (for continuous variable)
#input a dataset/sub-dataset(where last column is y), return the feature(column index) & cutpoint
def bestSplit(dataset):
    gini_before = Gini(colList(dataset,-1))
    gini_after = Gini(colList(dataset,-1))
    gless = 0
    #default value (using the smallest value of the first feature)
    small_1feat = min(colList(dataset,0))
    result = [0,small_1feat,gless]
    numFeature = len(dataset[0])-1
    #i: column index. iterate all features
    for i in range(0,numFeature):
        valueFeat = colList(dataset,i)
        uniqueV = set(valueFeat)
        valueFeat = list(uniqueV)
        valueFeat.sort()
        #iterate all values of a feature to find the best cutpoint
        for k in range(len(valueFeat)-1):
            mid = (valueFeat[k]+valueFeat[k+1])/2
            sub1,sub2 = splitData(dataset,i,mid)
            gini1 = Gini(colList(sub1,-1))
            gini2 = Gini(colList(sub2,-1))
            gini_after = len(sub1)/len(dataset)*gini1 + len(sub2)/len(dataset)*gini2
            if (gini_before - gini_after) > gless: 
                gless = gini_before - gini_after
                result = [i,mid,gless]
    return result[:-1]

#build decision tree. input a dataset, output tree(using dictionary)
def buildTree(dataset):
    yList=colList(dataset,-1)
    #all belong to the same class-> stop recursion
    if yList.count(yList[0]) == len(yList):
        return yList[0]
    #only one observation left -> stop recursion
    if len(dataset) == 1:
        return dataset[0][-1]

    feat = bestSplit(dataset)[0]
    cutvalue = bestSplit(dataset)[1]
    tree = {}
    tree['feature'] = feat
    tree['cutpoint'] = cutvalue
    leftS,rightS  = splitData(dataset,feat,cutvalue)
    tree['left'] = buildTree(leftS)
    tree['right'] = buildTree(rightS)
    return tree

#predict class of an observation. input tree and an observaion. 
def predict(dtree,obs):
    if obs[dtree['feature']] < dtree['cutpoint']:
        #base case for left-branch
        if isinstance(dtree['left'],int):
            return dtree['left']
        else:
            return predict(dtree['left'],obs)
    else: 
        #base case for right-branch
        if isinstance(dtree['right'],int):
            return dtree['right']
        else:
            return predict(dtree['right'],obs)

#predict a dataset. 
def predData(dtree,dataset):
    yhat = []
    for i in range(len(dataset)):
        label = predict(dtree,dataset[i])
        yhat.append(label)
    return yhat

#calculate accuracy.
def accuracy(true,pred):
    correct = 0
    for i in range(len(true)):
        if true[i] == pred[i]:
            correct += 1
    return correct/len(true)

#######################Random Forest##################################

#random return a number in range of (begin,end). To use it, call next(randomData (begin,end,seed)).
def randomData(begin,end,seed=999):
    a=32310901
    b=1729
    dataold=seed
    m=end-begin
    while True:
        datanew=(a*dataold+b)%m
        yield datanew
        dataold=datanew 

def randList(begin,end,num,seed=999):
    lst = []
    n = 0
    while n<= num:
        r = next(randomData(begin,end,n+seed)) #to make sure each time seed is different
        if r not in lst:
            lst.append(r)
            n += 1
    return lst

#best split for random forest. m: number of predictor candidates. seed: change 
#the difference between rfBestSplit() & bestSplit() is here feat_set is a subset of full predictors. 
def rfBestSplit(dataset,m,seed):
    gini_before = Gini(colList(dataset,-1))
    gini_after = Gini(colList(dataset,-1))
    gless = 0
    feat_set = randList(0,len(dataset[0])-1,m,seed)
    #default value (using the smallest value of the first feature)
    small_1feat = min(colList(dataset,feat_set[0]))
    result = [feat_set[0],small_1feat,gless]
   
    #i: column index. iterate features
    for i in feat_set:
        valueFeat = colList(dataset,i)
        # valueFeat = [x for x in valueFeat if valueFeat.count(x) == 1]
        uniqueV = set(valueFeat)
        valueFeat = list(uniqueV)
        valueFeat.sort()
        #iterate all values of a feature to find the best cutpoint
        for k in range(len(valueFeat)-1):
            mid = (valueFeat[k]+valueFeat[k+1])/2
            sub1,sub2 = splitData(dataset,i,mid)
            gini1 = Gini(colList(sub1,-1))
            gini2 = Gini(colList(sub2,-1))
            gini_after = len(sub1)/len(dataset)*gini1 + len(sub2)/len(dataset)*gini2
            if (gini_before - gini_after) > gless: 
                gless = gini_before - gini_after
                result = [i,mid,gless]
    return result[:-1]

#build a single tree for random forest. 
def buildRFTree(dataset,m,seed):
    yList=colList(dataset,-1)
    #all belong to the same class -> stop recursion
    if yList.count(yList[0]) == len(yList):
        return yList[0]
    #only one observation left -> stop recursion
    if len(dataset) == 1:
        return dataset[0][-1]

    feat = rfBestSplit(dataset,m,seed)[0]
    cutvalue = rfBestSplit(dataset,m,seed)[1]
    tree = {}
    tree['feature'] = feat
    tree['cutpoint'] = cutvalue
    leftS,rightS  = splitData(dataset,feat,cutvalue)
    tree['left'] = buildRFTree(leftS,m,seed)
    tree['right'] = buildRFTree(rightS,m,seed)
    return tree

#build random forest model(multiple trees). m: number of predictors of subset. n: n trees
def rf(dataset,m,n): 
    trees = []
    #build n decision trees
    for i in range(n):
        subset = []
        #random select observations as sub-dataset
        totalobs = len(colList(dataset,-1)) #total number of observations
        for k in range(totalobs):
            obsIndex = next(randomData(0,totalobs,99999+k))
            subset.append(dataset[obsIndex])
        atree = buildRFTree(subset,m,999998+i)
        trees.append(atree)
    return trees

#prediction for random forest. 
def rfPred(rforest,dataset):
    predlst = []
    result = []
    for i in range(len(rforest)):
        predlst.append(predData(rforest[i],dataset))
    #to final decide predicted class for an observation
    for j in range(len(dataset)):
        obspred = colList(predlst,j)
        if obspred.count(0) > obspred.count(1):
            result.append(0)
        else:
            result.append(1)
    return result

#####################Cross Validation################################

#Cross Validation split. 
# (here we did not randomly choose data to form a subdataset, we form subdatasets in order.)
#i: the ith as validation set, the other subsets as train dataset
def cvSplit(dataset,kfolds,i):
    split_sets = []
    num = len(dataset)//kfolds #number of obs in a fold
    for k in range(kfolds):
        fold = dataset[k*num:(k+1)*num]
        split_sets.append(fold)
    trn1 = split_sets
    valid = trn1.pop(i)
    trn = []
    for e in range(kfolds-1):
        for j in range(num):
            trn.append(trn1[e][j])
    return trn,valid

#cv for decision tree model. output estimated accuracy
def cvTree(dataset,kfolds):
    cvAcc = []
    for i in range(kfolds):
        train_set,valid_set = cvSplit(dataset,kfolds,i)
        tree = buildTree(train_set)
        pred = predData(tree,valid_set)
        truth = colList(valid_set,-1)
        cvAcc.append(accuracy(truth,pred))
    return sum(cvAcc)/len(cvAcc)

#cv for random forest model. output estimated accuracy. 
def cvRF(dataset,kfolds):
    cvAcc = []
    for i in range(kfolds):
        train_set,valid_set = cvSplit(dataset,kfolds,i)
        rfmodel = rf(train_set,3,11)
        pred = rfPred(rfmodel,valid_set)
        truth = colList(valid_set,-1)
        cvAcc.append(accuracy(truth,pred))
    return sum(cvAcc)/len(cvAcc)

###################################################################
