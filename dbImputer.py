import sqlite3
import numpy as np
import pandas
import sklearn
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.datasets
import sklearn.impute
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import csv

conn: sqlite3.Connection = sqlite3.connect('example.db')
cursor: sqlite3.Cursor = conn.cursor()

def doSql():
    print("Please enter SQL command line by line. End line with semicolon to finish.")
    userInput = input()
    fullCommand = ""
    while len(userInput) > 0 and userInput[-1] != ';':
        fullCommand += userInput + "\n"
        userInput = input()
    fullCommand += userInput
    result = cursor.execute(fullCommand).fetchall()
    ndResult = np.array(result)
    print(ndResult)
    conn.commit()
    return

def prepData(data: pandas.DataFrame, colName: str) -> tuple[np.ndarray, np.ndarray]:
    nullData = data[pandas.isna(data[colName])]
    notNullData = data.dropna(axis=0, subset=[colName])
    cols = data.columns.to_list()
    cols.remove(colName)
    for colName in data.columns.to_list():
        if pandas.api.types.is_datetime64_any_dtype(data[colName]):
            data[colName] = pandas.to_timedelta(data[colName]).dt.total_seconds()
    numDataExists = True
    strDataExists = True
    if len(data[cols].select_dtypes(include="number").columns) > 0:
        numImp = sklearn.impute.SimpleImputer()
        scaler = sklearn.preprocessing.StandardScaler()
        numNullData = scaler.fit_transform(numImp.fit_transform(nullData[cols].select_dtypes(include="number")))
        numNotNullData = scaler.fit_transform(numImp.fit_transform(notNullData[cols].select_dtypes(include="number")))
    else:
        numDataExists = False
    if len(data[cols].select_dtypes(exclude="number").columns) > 0:
        stringImp = sklearn.impute.SimpleImputer(strategy="most_frequent")
        strNullData = stringImp.fit_transform(nullData[cols].select_dtypes(exclude="number"))
        strNotNullData = stringImp.fit_transform(notNullData[cols].select_dtypes(exclude="number"))
        numberedStrNotNullData = None
        numberedStrNullData = None
        for col in strNotNullData.T:
            classSet = list(set(col))
            classSet.sort()
            if len(classSet) > 5:
                labelEncoder = sklearn.preprocessing.LabelEncoder()
                labelEncoder.fit(classSet)
                numberedCol = labelEncoder.transform(col.reshape(1, -1))
            else:
                oneHotEncoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
                classes = []
                for cName in classSet:
                    classes.append([cName])
                oneHotEncoder.fit(classes)
                numberedCol = oneHotEncoder.transform(col.reshape(-1, 1))
                print(numberedCol)
            if numberedStrNotNullData is not None:
                numberedStrNotNullData = np.concat(numberedStrNotNullData, numberedCol)
            else:
                numberedStrNotNullData = numberedCol
        for col in strNullData.T:
            classSet = list(set(col))
            classSet.sort()
            if len(classSet) > 5:
                labelEncoder = sklearn.preprocessing.LabelEncoder()
                labelEncoder.fit(classSet)
                numberedCol = labelEncoder.transform(col.reshape(1, -1))
            else:
                oneHotEncoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
                classes = []
                for cName in classSet:
                    classes.append([cName])
                oneHotEncoder.fit(classes)
                numberedCol = oneHotEncoder.transform(col.reshape(-1, 1))
                print(numberedCol)
            if numberedStrNullData is not None:
                numberedStrNullData = np.concat(numberedStrNullData, numberedCol)
            else:
                numberedStrNullData = numberedCol
    else:
        strDataExists = False
    if numDataExists and strDataExists:
        dataJoined = np.concat([numNotNullData, numberedStrNotNullData], axis=1)
        nullDataJoined = np.concat([numNullData, numberedStrNullData], axis=1)
    elif numDataExists:
        dataJoined = numNotNullData
        nullDataJoined = numNullData
    elif strDataExists:
        dataJoined = numberedStrNotNullData
        nullDataJoined = numNotNullData
    else:
        return None
    return (dataJoined, nullDataJoined)

def imputeData(model, preppedNull, table, tableName, colName, intFlag):
    preds = model.predict(preppedNull)
    predNum = 0
    for idx, entry in enumerate(table[colName]):
        if pandas.isna(entry):
            if intFlag:
                table.at[idx, colName] = round(preds[predNum])
                predNum+=1
            else:
                table.at[idx, colName] = preds[predNum]
                predNum+=1
    table.to_sql(tableName, conn, if_exists="replace", index=False)

def doImputation():
    print("Table names:")
    listOfTablesTups = cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'").fetchall()
    listOfTables = []
    for table in listOfTablesTups:
        listOfTables.append(table[0])
    print(listOfTables)
    userInput = input("Input table to impute.\n")
    if userInput not in listOfTables:
        print("Table not found.")
        return
    tableName = userInput
    listOfColsTups = cursor.execute("select name from pragma_table_info('" + userInput + "') as tblInfo;").fetchall()
    listOfCols = []
    for col in listOfColsTups:
        listOfCols.append(col[0])
    print("Columns:\n" + str(listOfCols))
    userInput = input("Select column to impute.\n")
    if userInput not in listOfCols:
        print("Column not found.")
        return
    colName = userInput

    table = pandas.read_sql_query(sql="SELECT * FROM \'" + tableName + "\';", con=conn, coerce_float=True)
    for idx, col in enumerate(listOfCols):
        table.rename(index={idx: col})
    print(table.dtypes)
    colType = table.dtypes[colName]
    otherCols = table.columns.to_list()
    otherCols.remove(colName)
    notNullData = table.dropna(axis=0, subset=[colName])
    if len(notNullData) == len(table):
        print("There are no missing values to imputate.")
        return
    preppedFeatures, preppedNullFeatures = prepData(table, colName)
    preppedLabels = notNullData[colName].to_numpy()

    bestModel = None
    bestModelChanged = False
    bestCvalAvg = 0
    bestCvals: np.ndarray

    classFlag = True
    if pandas.api.types.is_any_real_numeric_dtype(colType):
        classFlag = input("Should the numbers be treated as class labels?\n1. Yes\nOther. No\n") == "1"

    if not classFlag: #Regression
        intFlag = (input("Does the column store int values?\n1. Yes\nOther. No\n") == "1")

        print("\nBeginning model selection.\n")

        if len(preppedFeatures) > 10000:
            for i in [0.00001, 0.0001, 0.001]:
                print("Attempting SGD...")
                sgd = sklearn.linear_model.SGDRegressor(max_iter=1000, alpha=i, learning_rate='invscaling')
                sgd.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(sgd, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = sgd
                    bestCvals = cvals
            print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
            impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
            if impute:
                imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
        else:
            fewFeatures = input("Do you think that many of the other columns are unimportant?\n1. Yes\nOther. No\n")
            if fewFeatures:
                print("Attempting LASSO...")
                lassoReg = sklearn.linear_model.LassoCV(cv=5)
                lassoReg.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(lassoReg, preppedFeatures, preppedLabels)
                bestCvalAvg = np.mean(cvals)
                bestCvals = cvals
                bestModel = lassoReg
                print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
                    return
                
                print("Attempting Elastic Net...")
                elasticNet = sklearn.linear_model.ElasticNetCV(l1_ratio=0.7, cv=5)
                elasticNet.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(elasticNet, preppedFeatures, preppedLabels)
                if np.mean(cvals) > bestCvalAvg:
                    bestCvals = cvals
                    bestCvalAvg = np.mean(cvals)
                    bestModel = elasticNet
                    print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                    impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                    if impute:
                        imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
                        return
            else:
                print("Attempting Ridge Regression...")
                ridgeReg = sklearn.linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
                ridgeReg.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(ridgeReg, preppedFeatures, preppedLabels)
                bestCvals = cvals
                bestCvalAvg = np.mean(cvals)
                bestModel = ridgeReg
                print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
                    return
                bestModelChanged = False
                print("Attempting Linear SVR...")
                for i in [0.1, 1, 10]:
                    svrLinear = sklearn.svm.SVR(kernel='linear', C=i, max_iter=10000)
                    svrLinear.fit(preppedFeatures, preppedLabels)
                    cvals = sklearn.model_selection.cross_val_score(svrLinear, preppedFeatures, preppedLabels)
                    cvalAvg = np.mean(cvals)
                    if cvalAvg > bestCvalAvg:
                        bestCvalAvg = cvalAvg
                        bestModel = svrLinear
                        bestCvals = cvals
                        bestModelChanged = True
                if bestModelChanged:
                    print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                    impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                    if impute:
                        imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
                        return
                
                bestModelChanged = False
                print("Attempting RBF SVR...")
                for i in [0.1, 1, 10]:
                    svrRbf = sklearn.svm.SVR(kernel='rbf', C=i, max_iter=10000)
                    svrRbf.fit(preppedFeatures, preppedLabels)
                    cvals = sklearn.model_selection.cross_val_score(svrRbf, preppedFeatures, preppedLabels)
                    cvalAvg = np.mean(cvals)
                    if cvalAvg > bestCvalAvg:
                        bestCvalAvg = cvalAvg
                        bestModel = svrRbf
                        bestCvals = cvals
                        bestModelChanged = True
                if bestModelChanged:
                    print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                    impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                    if impute:
                        imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
                        return
                    
        useBest = input("All models have been attemped. Use the best found?\n1. Yes\nOther. No") == "1"
        if useBest:
            imputeData(bestModel, preppedNullFeatures, table, tableName, colName, intFlag)
        return
    
    else: #Classification

        print("\nBeginning model selection.\n")

        if len(preppedFeatures) > 10000:
            print("Attempting SGD classifier...")
            for i in [0.00001, 0.0001, 0.001]:
                sgdClass = sklearn.linear_model.SGDClassifier(max_iter=1000, alpha=i, learning_rate='invscaling')
                sgdClass.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(sgdClass, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = sgdClass
                    bestCvals = cvals
            print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
            impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
            if impute:
                imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
                return

            print("Attempting kernel classifier...")
            for i in [0.1, 1, 10]:
                kernel = sklearn.kernel_approximation.RBFSampler(gamma=i, random_state=1)
                kernel.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(kernel, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = kernel
                    bestCvals = cvals
                    bestModelChanged = True
            if bestModelChanged:
                print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
                    return
        else:
            print("Attempting Linear SVC classifier...")
            for i in [0.1, 1, 10]:
                linearSVC = sklearn.svm.LinearSVC(C=i)
                linearSVC.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(linearSVC, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = linearSVC
                    bestCvals = cvals
                    bestModelChanged = True
            if bestModelChanged:
                print("Cross Val Scores: " + str(bestCvals) + "\nAverage: " + str(bestCvalAvg) + "\n")
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
                    return
            bestModelChanged = False

            print("Attempting KNN classifier...")
            for i in [4, 8, 16]:
                knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
                knn.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(knn, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = knn
                    bestCvals = cvals
                    bestModelChanged = True
            if bestModelChanged:
                print("Cross Val Scores: " + str(bestCvals))
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
                    return
            bestModelChanged = False
            
            print("Attempting SVC classifier...")
            for i in [0.1, 1, 10]:
                svc = sklearn.svm.SVC(C=i, max_iter=10000)
                svc.fit(preppedFeatures, preppedLabels)
                cvals = sklearn.model_selection.cross_val_score(svc, preppedFeatures, preppedLabels)
                cvalAvg = np.mean(cvals)
                if cvalAvg > bestCvalAvg:
                    bestCvalAvg = cvalAvg
                    bestModel = svc
                    bestCvals = cvals
                    bestModelChanged = True
            if bestModelChanged:
                print("Cross Val Scores: " + str(bestCvals))
                impute = input("Is this acceptable for this data?\n1. Yes\nOther. No\n") == "1"
                if impute:
                    imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
                    return
                    
        useBest = input("All models have been attemped. Use the best found?\n1. Yes\nOther. No") == "1"
        if useBest:
            imputeData(bestModel, preppedNullFeatures, table, tableName, colName, False)
        return

def sklearnToTable(dataset: int):
    tableName = ""
    data: pandas.DataFrame
    if dataset == 0:
        tableName = "Diabetes"
        data = sklearn.datasets.load_diabetes(as_frame=True).frame
    elif dataset == 1:
        tableName = "Digits"
        data = sklearn.datasets.load_digits(as_frame=True).frame
    elif dataset == 2:
        tableName = "Wine"
        data = sklearn.datasets.load_wine(as_frame=True).frame
    elif dataset == 3:
        tableName = "Breast Cancer"
        data = sklearn.datasets.load_breast_cancer(as_frame=True).frame
    data.to_sql(name=tableName, if_exists="replace", con=conn, index=False)

def setUpExample():
    listOfTables = cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'").fetchall()
    for table in listOfTables:
        cursor.execute("DROP TABLE \'" + table[0] + "\'")
        conn.commit()
    userInput = input("Options:\n" + 
                      "1. Mendota/Monona Days\n" + 
                      "2. Iris Dataset\n" + 
                      "3. Diabetes\n" + 
                      "4. Digits\n" + 
                      "5. Wine\n" + 
                      "6. Breast Cancer\n")
    if userInput == "1":
        csvToTable("Ice Days", "./mendotaMonona.csv")
    elif userInput == "2":
        csvToTable("Iris", "./iris-3.csv")
    elif userInput == "3":
        sklearnToTable(0)
    elif userInput == "4":
        sklearnToTable(1)
    elif userInput == "5":
        sklearnToTable(2)
    elif userInput == "6":
        sklearnToTable(3)
    else:
        print("Option not recognized.")
    return

def deleteFromCol():
    print("Table names:")
    listOfTablesTups = cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'").fetchall()
    listOfTables = []
    for table in listOfTablesTups:
        listOfTables.append(table[0])
    print(listOfTables)
    userInput = input("Input table to delete from.\n")
    if userInput not in listOfTables:
        print("Table not found.")
        return
    tableName = userInput
    
    listOfColsTups = cursor.execute("select name from pragma_table_info('" + tableName + "') as tblInfo;").fetchall()
    listOfCols = []
    for col in listOfColsTups:
        listOfCols.append(col[0])
    print("Columns:\n" + str(listOfCols))
    userInput = input("Select column to delete from.\n")
    if userInput not in listOfCols:
        print("Column not found.")
        return
    colName = userInput

    table = pandas.read_sql_query(sql="SELECT * FROM \'" + tableName + "\';", con=conn, coerce_float=True)
    for idx, col in enumerate(listOfCols):
        table.rename(index={idx: col})
    print(table)
    print(table.dtypes)
    print(colName)
    userInput = input("Should deletion use a seed?\n1. Yes\nOther. No\n")
    gen = np.random.default_rng()
    if userInput == "1":
        userInput = input("Input seed (must be an int).\n")
        seed = 0
        try:
            seed = int(userInput)
        except:
            print("Please use an int.")
            return
        gen = np.random.default_rng(seed)
    
    #userInput = input("How should data be deleted? ONLY 3 IMPLEMENTED CURRENTLY\n1. Missing at random\n2. Missing not at random\n3. Missing completely at random\n")
    #if userInput == "1":
    #    print()
    #elif userInput == "2":
    #    print()
    #elif userInput == "3":
    table[colName] = table.apply(lambda x: x[colName] if gen.random() < 0.9 else np.nan, axis=1)
    #else:
    #    print("Option not recognized.")
    
    table.to_sql(name=tableName, if_exists="replace", con=conn, index=False)
    print(table)
    print(table.dtypes)
    conn.commit()
    return

def csvToTable(tableName: str, filePath: str):
    with open(filePath, newline='', encoding="utf8") as csvfile:
        data = pandas.read_csv(csvfile)
        print(data)
        print(data.dtypes)
        for name, vals in data.items():
            if pandas.api.types.is_string_dtype(vals):
                try:
                    dateVals = pandas.to_datetime(vals, errors="raise", format="mixed")
                except:
                    pass
                else:
                    data[name] = dateVals
        print(data)
        print(data.dtypes)
        data.to_sql(name=tableName, if_exists="replace", con=conn, index=False)
    return

def csvToTableUser():
    tableName = input("Please enter table name. If table already exists, it will be dropped.\n")
    filePath = input("Please enter CSV file path. .csv should be included.\n")
    csvToTable(tableName, filePath)
    return

def main():
    queryString = ("Please enter command.\n"
                   + "1. Perform SQL query\n"
                   + "2. Perform imputation\n"
                   + "3. Set up example DB\n"
                   + "4. CSV to table (dates can be finnicky)\n"
                   + "5. Delete data from column\n"
                   + "6. Exit\n")
    userInput = input(queryString)
    while userInput != "6":
        if userInput == "1":
            doSql()
        elif userInput == "2":
            doImputation()
        elif userInput == "3":
            setUpExample()
        elif userInput == "4":
            csvToTableUser()
        elif userInput == "5":
            deleteFromCol()
        else:
            print("Command not recognized.\n")
        userInput = input(queryString)
    print("Exiting database.")
    conn.commit()
    conn.close()

main()