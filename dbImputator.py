import pandas
import sqlite3
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.impute
import sklearn.preprocessing
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
    nullData = data[data[colName] is None or np.isnan(data[colName])]
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
        numNullData = numImp.fit_transform(nullData[cols].select_dtypes(include="number"))
        numNotNullData = numImp.fit_transform(notNullData[cols].select_dtypes(include="number"))
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

def doImputation():
    print("Table names:")
    listOfTablesTups = cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'").fetchall()
    listOfTables = []
    for table in listOfTablesTups:
        listOfTables.append(table[0])
    print(listOfTables)
    userInput = input("Input table to imputate.\n")
    if userInput not in listOfTables:
        print("Table not found.")
        return
    tableName = userInput
    listOfColsTups = cursor.execute("select name from pragma_table_info('" + userInput + "') as tblInfo;").fetchall()
    listOfCols = []
    for col in listOfColsTups:
        listOfCols.append(col[0])
    print("Columns:\n" + str(listOfCols))
    userInput = input("Select column to imputate.\n")
    if userInput not in listOfCols:
        print("Column not found.")
        return
    colName = userInput

    table = pandas.read_sql_query(sql="SELECT * FROM " + tableName + ";", con=conn, coerce_float=True)
    for idx, col in enumerate(listOfCols):
        table.rename(index={idx: col})
    print(table.dtypes)
    colType = table.dtypes[colName]
    otherCols = table.columns.to_list()
    otherCols.remove(colName)
    notNullData = table.dropna(axis=0, subset=[colName])
    preppedData, preppedNullData = prepData(table, colName)
    if pandas.api.types.is_any_real_numeric_dtype(colType):
        forestClf = sklearn.ensemble.RandomForestRegressor(n_estimators=16)
        #print(preppedData)
        forestClf.fit(preppedData, notNullData[colName].to_numpy())
        preds = forestClf.predict(preppedNullData)
        #table.to_sql(tableName, conn, if_exists="replace", index=False)
        print(preds)
        return
    else:
        forestClf = sklearn.ensemble.RandomForestClassifier(n_estimators=16)
        #print(preppedData)
        forestClf.fit(preppedData, notNullData[colName].to_numpy())
        preds = forestClf.predict(preppedNullData)
        #table.to_sql(tableName, conn, if_exists="replace", index=False)
        print(preds)
        return
    return

def setUpExample():
    listOfTables = cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'").fetchall()
    for table in listOfTables:
        cursor.execute("DROP TABLE \'" + table[0] + "\'")
        conn.commit()
    userInput = input("Options:\n1. Mendota/Monona Days\n2. Iris Dataset\n")
    if userInput == "1":
        csvToTable("Ice Days", "./mendotaMonona.csv")
    elif userInput == "2":
        csvToTable("Iris", "./iris-3.csv")
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

    table = pandas.read_sql_query(sql="SELECT * FROM " + tableName + ";", con=conn, coerce_float=True)
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
    
    userInput = input("How should data be deleted? ONLY 3 IMPLEMENTED CURRENTLY\n1. Missing at random\n2. Missing not at random\n3. Missing completely at random\n")
    if userInput == "1":
        print()
    elif userInput == "2":
        print()
    elif userInput == "3":
        table[colName] = table.apply(lambda x: x[colName] if gen.random() < 0.9 else np.nan, axis=1)
    else:
        print("Option not recognized.")
    
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