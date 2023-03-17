from   sklearn.preprocessing  import LabelEncoder

# checking for the NAN values in the dataset
def check_nan_cols(data):
  nan_columns = [i for i in data.columns if data[i].isnull().sum() != 0]
  print(f'Features with null values = {nan_columns}')

def label_encode(new_feature, old_feature,data):
  data[new_feature] = LabelEncoder().fit_transform(data[old_feature])
  return data
  
def create_img_path(feature, data, path):
  data[feature] = data[feature].apply(lambda x: path+x+'.png')
  return data


def get_class_numbers(data):
  return data['target'].nunique()