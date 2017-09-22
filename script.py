import os
from data import fetch_data
a,b,c,d = fetch_data('',remove_duplicates=False,binary=True)
from data import feature_engineering
x = feature_engineering(a,do_normalization=True)
os.chdir("../")
from aug import Sampling
s = Sampling(x,b)
new_data = s.Smote_Over_Samp(x,b)
