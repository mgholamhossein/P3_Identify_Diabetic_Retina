# P3_Identify_Diabetic_Retina
Assessing diabetic severity using retina images

# setting up the project after cloing the project on your local computer
1- Modify Datasetcfg() class in "P3_Identify_Diabetic_Retina.ipynb" file. 
check all the path in this class and make sure these paths exist on your computer.
update the model name you wish to save later in training part appropriately to avoid overwriting your previously saved models.
![image](https://user-images.githubusercontent.com/65259199/155895762-09ce8ca3-859c-4d63-8373-0865c7eb0a62.png)


2- Update your project configuation in Config.py file
![image](https://user-images.githubusercontent.com/65259199/155895845-4bce0ca2-f182-4f5e-8ac1-25167cc45c1d.png)


3- Update number of epochs in the train_epochs() parameters as needed in "P3_Identify_Diabetic_Retina.ipynb" file. 
modify the model you are planning to use here.
TIP: For quick runs, you can use one fold of data by using range(1) instead of range(Config.kfolds) before you call the training module.
![image](https://user-images.githubusercontent.com/65259199/155895881-bba9afb6-3bd3-4d30-8d9e-eac24a72c7b7.png)
