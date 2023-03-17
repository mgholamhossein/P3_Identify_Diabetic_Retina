import random
import cv2
import matplotlib.pyplot as plt
random.seed(100)

# test if these data loaders work well
def verify_dataloader(data_loader_be_tested, lines=5):
  for step, (batch_images, batch_labels) in enumerate(data_loader_be_tested):
    if step>(lines-1):break # for purpose of testing, only run 4 batches
    print(batch_images.shape,batch_labels.shape)


def check_GPU(gpu_info):
  gpu_info = '\n'.join(gpu_info)
  if gpu_info.find('failed') >= 0:
    print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
    print('and then re-execute this cell.')
  else:
    print(gpu_info)

# pick up k different label groups, in each label group, pick up m images randomly and plot them together
def plot_rand_images(data,path_feature,target_feature,k,m):
  
  temp = data[target_feature].value_counts()
  print(temp)
  x_list = list(temp[temp > k].index)
  # randomly select the k label_groups that has more than 5 images in them
  label_group_list = random.sample(x_list, k)

  # select m images from 
  img_list=[]
  for i in label_group_list:
    temp_df = data[data[target_feature] == i][path_feature]
    temp_df = temp_df.sample(frac = 1)
    img_list.extend(temp_df[:m])
    print(img_list)
  row = k
  col = m
  figure = plt.figure(figsize = (30,30))
  for i in range(row*col):
    plt.subplot(row,col,i+1)
    img = cv2.imread(img_list[i]) 
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'image size: {img.shape}')

  plt.show()