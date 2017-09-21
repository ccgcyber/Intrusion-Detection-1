import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def paralearn(name,model,x_train,y_train,nfolds=3):
    if name=='svm':
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(x_train,y_train)
        grid_search.best_params_
        return grid_search.best_params_
    elif name=='randforest':
        model=RandomForestClassifier(criterion='entropy',max_depth=20,
                                     random_state=12345,n_jobs=2)
        param_grid = {
                'n_estimators': [200, 700],
                'max_features': ['auto', 'sqrt', 'log2']
                }
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
        CV_rfc.fit(x_train, y_train)
        return CV_rfc.best_params_
    elif name=='xgboost':
        x_dtrain=xgb.DMatrix(data=x_train,labels=y_train)
        x_dtest=xgb.DMatrix(data=x_test)
        param_test1 = {
                'max_depth':range(3,10,2),
                'min_child_weight':range(1,6,2)
                }
        gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                          min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                          objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
        gsearch1.fit(train[predictors],train[target])       
        return gsearch1.best_params_
    elif name=='mnbayes':
        return
    else:
        print('invalid model input')
        return
        
        

