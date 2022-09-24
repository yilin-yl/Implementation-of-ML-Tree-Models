To use the source code, src.py, you should import src.  If you want to use the functions written in src.py, you can call like src.f(x). 

We also wrote main.py as entry file. 
To run the main.py, you will be firstly asked to enter the train dataset file and test dataset file as well as their directory. 
For example,  when the screen shows "Enter train.csv and its directory:", you may enter C:\Users\xxx\Documents\train.csv 
After read 2 csv file, main.py will print out 3 accuracy results: 
1. (CV)estimated accuracy of CART, 
2. (CV)estimated test accuracy of random forest model, 
3. test accuracy of our final model- random forest. 
These may take around 2 minutes. 
