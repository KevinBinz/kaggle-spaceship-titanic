import seaborn as sns
import matplotlib.pyplot as plt


def getAllCombos(labelList):
    comboList = []
    for i in range(len(labelList)):
        for j in range(i+1, len(labelList)):
            comboList.append((labelList[i], labelList[j]))
    return comboList


def numCombo_ScatterPlots(df, comboList, colorLabel):
    numCombos = len(comboList)
    fig=plt.figure(figsize=(10,numCombos*4))
    for i, (x, y) in enumerate(comboList):
        ax=fig.add_subplot(numCombos,1,i+1)
        sns.scatterplot(data=df, x=x, y=y, hue=colorLabel)
        ax.set_title(f'{x} vs {y}') 
    fig.tight_layout()

def numCombo_Histograms(df, numericList, colorLabel):
    numNumerics = len(numericList)
    fig=plt.figure(figsize=(10,numNumerics*4))
    for i, numeric in enumerate(numericList):
        ax=fig.add_subplot(numNumerics,1,i+1)
        sns.histplot(data=df, x=numeric, hue=colorLabel, kde=True)
        ax.set_title(numeric) 
    fig.tight_layout()

def catSolo_CountPlots(df, categoricalList, colorLabel):
    numFeats = len(categoricalList)
    fig=plt.figure(figsize=(10,numFeats*4))
    for i, var_name in enumerate(categoricalList):
        ax=fig.add_subplot(numFeats,1,i+1)
        if colorLabel == None:
            sns.countplot(data=df, x=var_name, axes=ax)
        else:
            sns.countplot(data=df, x=var_name, axes=ax, hue=colorLabel)
        ax.set_title(var_name) 
    fig.tight_layout()

def catCombo_GbCountplot(df, comboList):
    dfDict = {}
    for tpl in comboList:
        (y,x) = tpl
        catCountDF = df.groupby([y, x])[x].size().unstack().fillna(0)
        uniqueSeries = (catCountDF>0).sum(axis=1).reset_index(name='Count')['Count'].astype(str)
        dfDict[tpl] = uniqueSeries
    numCombos = len(comboList)

    fig=plt.figure(figsize=(10,numCombos*4))
    for i, (tpl, dfgb) in enumerate(dfDict.items()):
        (x,y) = tpl
        ax=fig.add_subplot(numCombos,1,i+1)
        sns.countplot(x=dfgb)
        ax.set_title(f"Unique {y} per {x}")
    fig.tight_layout()

def catCombo_Heatmap(df, comboList):
    numCombos = len(comboList)
    fig=plt.figure(figsize=(10,numCombos*4))
    for i, (x, y) in enumerate(comboList):
        gbdf=df.groupby([y, x])[x].size().unstack().fillna(0)
        ax=fig.add_subplot(numCombos,1,i+1)
        sns.heatmap(gbdf.T, annot=True, fmt='g', cmap='coolwarm')
        ax.set_title(f'{x} vs {y}') 
    fig.tight_layout()