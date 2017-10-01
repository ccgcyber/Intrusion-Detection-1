import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as RUS


class Sampling:

  def __init__(self,X_t,Y_t):
    self.X_t = X_t
    self.Y_t = Y_t

  def Rand_Over_Samp(self,X_t,Y_t):
    X_train = pd.DataFrame(self.X_t)
    Y_train = pd.DataFrame(self.Y_t)
    comb = pd.concat([X_train,Y_train],axis=1)
    l = list(comb)
    sampler = ROS(random_state=42)
    sampled_X,sampled_Y = sampler.fit_sample(X_train,Y_train.values.ravel())
    sampled_X = pd.DataFrame(sampled_X)
    sampled_Y = pd.DataFrame(sampled_Y)
    data_for_modelling = np.concatenate([sampled_X,sampled_Y],axis=1)
    data_for_modelling = pd.DataFrame(data_for_modelling)
    data_for_modelling.columns = l
    return data_for_modelling

  def Smote_Over_Samp(self,X_t,Y_t):
    X_train = pd.DataFrame(self.X_t)
    Y_train = pd.DataFrame(self.Y_t)
    comb = pd.concat([X_train,Y_train],axis=1)
    l = list(comb)
    sampler = SMOTE(kind="borderline1",ratio=1)
    sampled_X,sampled_Y = sampler.fit_sample(X_train,Y_train.values.ravel())
    sampled_X = pd.DataFrame(sampled_X)
    sampled_Y = pd.DataFrame(sampled_Y)
    data_for_modelling = np.concatenate([sampled_X,sampled_Y],axis=1)
    data_for_modelling = pd.DataFrame(data_for_modelling)
    data_for_modelling.columns = l
    return data_for_modelling

  def Rand_Under_Samp(self,X_t,Y_t):
    X_train = pd.DataFrame(self.X_t)
    Y_train = pd.DataFrame(self.Y_t)
    comb = pd.concat([X_train,Y_train],axis=1)
    l = list(comb)
    sampler = RUS(random_state=42)
    sampled_X,sampled_Y = sampler.fit_sample(X_train,Y_train.values.ravel())
    sampled_X = pd.DataFrame(sampled_X)
    sampled_Y = pd.DataFrame(sampled_Y)
    data_for_modelling = np.concatenate([sampled_X,sampled_Y],axis=1)
    data_for_modelling = pd.DataFrame(data_for_modelling)
    data_for_modelling.columns = l
    return data_for_modelling

