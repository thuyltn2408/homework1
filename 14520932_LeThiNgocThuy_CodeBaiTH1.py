
# coding: utf-8

# # Bài thực hành số 1
# 

# # Mục lục
# ## <a href='#1'>Bài tập 1</a>
# ## <a href='#2'>Bài tập 2</a>
# ### <a href='#2.1'>2.1 Load digits dataset</a>
# ### <a href='#2.2'>2.2 Kmeans </a>
# ### <a href='#2.3'>2.3 Spectral</a>
# ### <a href='#2.4'>2.4 DBSCAN</a>
# ### <a href='#2.5'>2.5 Agglomerative</a>
# ### <a href='#2.6'>2.6 Visualize</a>
# ### <a href='#2.7'>2.7 Evaluation</a>
# #### <a href='#2.7.1'>2.7.1 adjusted_mutual_info_score</a>
# #### <a href='#2.7.2'>2.7.2 mutual_info_score</a>
# #### <a href='#2.7.3'>2.7.3 homogeneity_completeness_v_measure</a>
# ### <a href='#2.8'>2.8 Nhận xét</a>
# ## <a href='#3'>Bài tập 3</a>
# ### <a href='#3.1'>Load Faces dataset</a>
# ### <a href='#3.2'>LBP feature</a>
# ### <a href='#3.3'>3.3 KMeans</a>
# ### <a href='#3.4'>3.4 Specctral</a>
# ### <a href='#3.5'>3.5 DBSCAN</a>
# ### <a href='#3.6'>3.6 Agglomerative</a>
# ### <a href='#3.7'>3.7 Visualize</a>
# ### <a href='#3.8'>3.8 Evaluation</a>
# ### <a href='#3.9'>3.9 Nhận xét</a>
# ## <a href='#4'>Bài tập 4</a>
# ### <a href='#4.1'>4.1 Load Car dataset</a>
# ### <a href='#4.2'>4.2 Rút trích HoG feature</a>

# ### Github: https://github.com/thuyltn2408/homework1

# <a id='1'></a>
# # Bài tập 1: KMeans trên tập dữ liệu ngẫu nhiên
# 

# ** Bước 1: Phát sinh dữ liệu
#         - Sử dụng sklearn.dataset.make_bobs để phát sinh dữ liệu ngẫu nhiên với số cluster là 2
#         - Visualize dữ liệu đã phát sinh
# ** Bước 2: Áp dụng KMeans
#         - Dùng sklearn.cluster.KMeans để clustering dữ liệu trên với số cluster bằng 2
# ** Bước 3: Visualize
#         - Dùng pandas.scatter để visualize dữ liệu và centerPoints của clusters

# In[119]:


# Import thư viện
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.gray()


# In[120]:


# Bước 1
#      1.1: Phát sinh dữ liệu với 150 mẫu, số lớp: 2
X,Y = make_blobs(n_samples=150, n_features=2, centers=2)


# In[121]:


#      1.2: Visualize dữ liệu
plt.scatter(X[:,0],X[:,1])
plt.show()


# In[122]:


# Bước 2: Áp dụng KMeans clustering trên dữ liệu đã phát sinh
model = KMeans(n_clusters=2)
labels1 = model.fit_predict(X)


# In[123]:


# Bước 3: Visualize kết quả với tập dữ liệu và centerPoints của cluster
plt.scatter(X[:,0],X[:,1], c=labels1 ,alpha=1)

center_points=model.cluster_centers_
plt.scatter(center_points[:,0],center_points[:,1] ,marker='D',s=50)

plt.show()


# <a id='2'></a>
# # Bài tập 2: KMeans, Spectral, DBSCAN, Agglomerative clustering trên Digits data

# - 2.1 Load dataset: digits data
# - 2.2 KMeans clustering
# - 2.3 Spectral clustering
# - 2.4 DBSCAN clustering
# - 2.5 Agglomerative clustering
# - 2.6 Visualize
# - 2.7 Evaluation
# - 2.8 Nhận xét

# In[124]:


# Import thư viện
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import image
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN


# <a id='2.1'></a>
# ### 2.1 Load dataset

# In[125]:


# Load dataset
digits = datasets.load_digits()
print(digits.data.shape)


# <a id='2.2'></a>
# ### 2.2 KMeans trên dữ liệu

# In[126]:


# KMeans trên digits data
model2 = KMeans(n_clusters=10)
labels = model2.fit_predict(digits.data)


# - Thống kê kết quả
# - Cột đầu tiên: nhãn mà KMeans clustering đã gán vào dữ liệu
# - Hàng đầu tiền: nhãn đúng của dữ liệu
# - Ý nghĩa phần tử hàng i, cột j , i và j > 0, phần tử có giá trị là value:
#     - Có value ảnh mà clustering gán nhãn i có nhãn thật sự là j

# In[127]:


df=pd.DataFrame({'labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['labels'],df['Truth labels'])
print(ct)


# In[128]:


# Kiểm tra ảnh thứ n nhãn clustering đã gán có khớp với nhãn đúng hay không.
n=15
plt.matshow(digits.images[n])
print('lables_predict:',labels[n])
print(' True: ', digits.target[n])


# <a id='2.3'></a>
# ### 2.3 Spectral clustering

# In[201]:


# Tính ma trận độ tương đồng của data
graph = cosine_similarity(digits.data)

# Áp dụng Spectral clustering cho data
labels_spectral = spectral_clustering(graph, n_clusters=10)


# In[130]:


# Thống kê kết quả
df1=pd.DataFrame({'labels':labels_spectral,'Truth labels':digits.target})
ct2=pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# In[202]:


# Kiểm tra kết quả
n=10
plt.matshow(digits.images[n])
print('lables_predict:',labels_spectral[n])
print(' True: ', digits.target[n])


# <a id='2.4'></a>
# ### 2.4 DBSCAN clustering

# In[132]:


# Gán thông số cho DBSCAN clustering
eps = 0.0595
min_samples = 10

# Áp dụng DBSCAN cho data
dbscan= DBSCAN(eps=eps, min_samples =min_samples,metric='cosine')
labels_dbscan = dbscan.fit_predict(digits.data)

# Thống kê kết quả
df_dbscan = pd.DataFrame({'labels':labels_dbscan,'Truth labels':digits.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# In[133]:


# Kiểm tra kết quả
n= 10
plt.matshow(digits.images[n])
print('lables_predict:',labels_dbscan[n])
print(' True: ', digits.target[n])


# <a id='2.5'></a>
# ### 2.5 AgglomerativeClustering

# In[203]:


# Áp dụng Agglomerative clustering
model = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean')
labels_Agg = model.fit_predict(digits.data)

# Thống kê kết quả
df=pd.DataFrame({'labels':labels_Agg,'Truth labels':digits.target})
ct = pd.crosstab(df['labels'],df['Truth labels'])
print(ct)


# In[204]:


# Kiểm tra kết quả
n= 7
plt.matshow(digits.images[n])
print('lables_predict:',labels_Agg[n])
print(' True: ', digits.target[n])


# <a id='2.6'></a>
# ### 2.6 Visualize

# In[136]:


# Sử dụng PCA để convert dữ liệu về 2 chiều để visualize kết quả
pca = PCA(n_components = 2)
digitsData_to_2dimention = pca.fit_transform(digits.data)

# Tạo figure để visualize kết quả
Figure = plt.figure(figsize=(12,12))


fi = Figure.add_subplot(3, 2, 3)
# Visualize kết quả với nhãn đúng
fi.scatter(digitsData_to_2dimention[:,0], digitsData_to_2dimention[:,1],  c= digits.target, s=20)
fi.set_title('True labels')


fi = Figure.add_subplot(3, 2, 1)
# Visualize kết quả với nhãn dùng KMeans cluster
fi.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels,s=20)
fi.set_title('KMeans')


fi = Figure.add_subplot(3, 2, 2)
# Visualize kết quả với nhãn dùng spectral cluster
fi.scatter(digitsData_to_2dimention[:,0], digitsData_to_2dimention[:,1],  c= labels_spectral, s=20)
fi.set_title('Spectral')


fi = Figure.add_subplot(3, 2, 5)
# Visualize kết quả với nhãn dùng DBSCAN
plt.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels_dbscan,s=20)
fi.set_title('DBSCAN')

fi = Figure.add_subplot(3, 2, 6)
# Visualize kết quả với nhãn dùng AgglomerativeClustering
plt.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels_Agg,s=20)
fi.set_title('AgglomerativeClustering')


# <a id='2.7'></a>
# ### 2.7 Evaluation

# - Sử dụng sklearn.metrics để đánh giá kết quả của các phương pháp clustering trên tập dữ liệu là các ảnh số viết tay (Digits data)

# In[137]:


# Import thư viện để đánh giá kết quả
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure


# <a id='2.7.1'></a>
# #### 2.7.1 Thực hiện đáng giá theo adjusted_mutual_info_score

# In[138]:


# KMeans
print("KMeans evaluation: ",adjusted_mutual_info_score(digits.target, labels))

# Spectral cluster
print("Spectral evaluation: ",adjusted_mutual_info_score(digits.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",adjusted_mutual_info_score(digits.target, labels_dbscan))

# AgglomerativeClustering
print("AgglomerativeClustering evaluation: ",adjusted_mutual_info_score(digits.target, labels_Agg))


# <a id='2.7.2'></a>
# #### 2.7.2 Thực hiện đáng giá theo mutual_info_score

# In[139]:


# KMeans
print("KMeans evaluation: ",mutual_info_score(digits.target, labels))

# Spectral cluster
print("Spectral evaluation: ",mutual_info_score(digits.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",mutual_info_score(digits.target, labels_dbscan))

# AgglomerativeClustering
print("AgglomerativeClustering evaluation: ",mutual_info_score(digits.target, labels_Agg))


# <a id='2.7.3'></a>
# #### 2.7.3 Thực hiện đáng giá theo homogeneity_completeness_v_measure
# - Giá trị trả về trong khoảng 0 >> 1
# - Càng về 1 thì độ khớp của True labels và cluster labels càng cao.

# In[140]:


# KMeans
print("KMeans evaluation: ",homogeneity_completeness_v_measure(digits.target, labels))

# Spectral cluster
print("Spectral evaluation: ",homogeneity_completeness_v_measure(digits.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",homogeneity_completeness_v_measure(digits.target, labels_dbscan))

# AgglomerativeClustering
print("AgglomerativeClustering evaluation: ",homogeneity_completeness_v_measure(digits.target, labels_Agg))


# <a id='2.8'></a>
# ### 2.8 Nhận xét
# - Đối với data là chữ số viết tay (Digits data) thì Agglomerative clustering hiệu quả hơn hẳn so với KMeans, Spectral, DBSCAN clustering
# - DBSCAN clustering: khó sử dụng bởi parameters: eps và min_samples. Thử nhiều lần giá trị của eps và min_sample mới cho kết quả khả quan.

# <a id='3'></a>
# # Bài tập 3:  KMeans, Spectral, DBSCAN, Agglomerative clustering trên Faces data

# - 3.1 Load dataset: Faces data
# - 3.2 Rút trích LBP feature
# - 3.3 KMeans clustering
# - 3.4 Spectral clustering
# - 3.5 DBSCAN clustering
# - 3.6 Agglomerative clustering
# - 3.7 Visualize
# - 3.8 Evaluation
# - 3.9 Nhận xét

# <a id='3.1'></a>
# ### 3.1 Load Faces datsaset

# In[141]:


# Import thư viện
from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import numpy as np

# Load Faces dataset
# Chỉ load ảnh khi nhóm của ảnh đó có ít nhất 80 ảnh
faces_data = fetch_lfw_people(min_faces_per_person=80)
print("Number of images: ", faces_data.images.shape)
print("Number of classes: ",len(set(faces_data.target)))


# In[142]:


# create a figure to show image
Figure = plt.figure(figsize=(8,5))

# for all 0-9 labels
for i in range(100,110):
    # initialize subplots in a grid 2x5 at i+1th position
    fi = Figure.add_subplot(2, 5, 1+i-100)
    
    # display image
    fi.imshow(faces_data.images[i], cmap=plt.cm.binary)
    
    #don't show the axes
    plt.axis('off')

plt.show()


# <a id='3.2'></a>
# ### 3.2 Rút trích feature LBP

# In[143]:


# Rút trích LBP feature cho ảnh đầu tiên của dataset
featureLBP = local_binary_pattern(faces_data.images[0], P=8, R=0.5)


# In[144]:


plt.matshow(featureLBP)


# In[145]:


print(featureLBP)


# In[146]:


Figure = plt.figure(figsize=(15,5))
fi = Figure.add_subplot(1,1,1)
fi.hist(featureLBP.reshape(-1), bins=list(range(257)))
plt.title('256-dimensional feature vector')
plt.show()


# In[147]:


# Xây dựng hàm rút trích LBP feature cho 1 ảnh
def extractFeatureLBP(image):
    featureLBP = local_binary_pattern(image, P=8, R=0.5)
    return np.histogram(featureLBP, bins=list(range(257)))[0]


# In[148]:


# Rút trích feature LBP trên tập data
featureLBP = list(map(extractFeatureLBP, faces_data.images))


# In[149]:


featureLBP = np.array(featureLBP)
type(featureLBP)
print(featureLBP.shape)
print(featureLBP[0:2])


# <a id='3.3'></a>
# ### 3.3 KMeans cluster

# In[150]:


# Áp dụng KMeans trên Faces dataset
# Với dữ liệu đầu vào là LBP feature
model = KMeans(n_clusters=5)
labels = model.fit_predict(featureLBP)
print(labels)

# Thống kê kết quả
df = pd.DataFrame({'label':labels, 'True Label':faces_data.target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct.tail(10))


# <a id='3.4'></a>
# ### 3.4 Spectral cluster

# In[151]:


# Tính ma trận độ tương đồng của data
graph = cosine_similarity(featureLBP)

# Áp dụng Spectral clustering cho data
labels_spectral = spectral_clustering(graph, n_clusters=5)

# Thống kê kết quả
df1=pd.DataFrame({'labels':labels_spectral,'Truth labels':faces_data.target})
ct2=pd.crosstab(df1['labels'],df1['Truth labels'])
print(ct2)


# <a id='3.5'></a>
# ### 3.5 DBSCAN

# In[152]:


# Gán thông số cho cluster
eps = 50
min_samples = 10

# Áp dụng DBSCAN cho data
dbscan= DBSCAN(eps=eps, min_samples =min_samples,metric='cosine')
labels_dbscan = dbscan.fit_predict(featureLBP)

# Thống kê kết quả
df_dbscan = pd.DataFrame({'labels':labels_dbscan,'Truth labels':faces_data.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# <a id='3.6'></a>
# ### 3.6 AgglomerativeClustering

# In[153]:


model = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean')
labels_Agg = model.fit_predict(featureLBP)

# Thống kê kết quả
df=pd.DataFrame({'labels':labels_Agg,'Truth labels':faces_data.target})
ct = pd.crosstab(df['labels'],df['Truth labels'])
print(ct)


# <a id='3.7'></a>
# ### 3.7 Visualize

# In[154]:


# Sử dụng PCA để convert dữ liệu về 2 chiều để visualize kết quả
pca = PCA(n_components = 2)
digitsData_to_2dimention = pca.fit_transform(featureLBP)


# Tạo figure để visualize kết quả
Figure = plt.figure(figsize=(12,12))


fi = Figure.add_subplot(3, 2, 3)
# Visualize kết quả với nhãn đúng
fi.scatter(digitsData_to_2dimention[:,0], digitsData_to_2dimention[:,1],  c= faces_data.target, s=20)
fi.set_title('True labels')


fi = Figure.add_subplot(3, 2, 1)
# Visualize kết quả với nhãn dùng KMeans cluster
fi.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels,s=20)
fi.set_title('KMeans')


fi = Figure.add_subplot(3, 2, 2)
# Visualize kết quả với nhãn dùng spectral cluster
fi.scatter(digitsData_to_2dimention[:,0], digitsData_to_2dimention[:,1],  c= labels_spectral, s=20)
fi.set_title('Spectral')


fi = Figure.add_subplot(3, 2, 5)
# Visualize kết quả với nhãn dùng DBSCAN
plt.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels_dbscan,s=20)
fi.set_title('DBSCAN')

fi = Figure.add_subplot(3, 2, 6)
# Visualize kết quả với nhãn dùng AgglomerativeClustering
plt.scatter(digitsData_to_2dimention[:,0],digitsData_to_2dimention[:,1], c=labels_Agg,s=20)
fi.set_title('AgglomerativeClustering')


# <a id='3.8'></a>
# ### 3.8 Evaluation

# In[155]:


#Thực hiện đáng giá theo adjusted_mutual_info_score

# KMeans
print("KMeans evaluation: ",adjusted_mutual_info_score(faces_data.target, labels))

# Spectral cluster
print("Spectral evaluation: ",adjusted_mutual_info_score(faces_data.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",adjusted_mutual_info_score(faces_data.target, labels_dbscan))

# AgglomerativeClustering
print("AgglomerativeClustering evaluation: ",adjusted_mutual_info_score(faces_data.target, labels_Agg))


# In[156]:


# Thực hiện đáng giá theo mutual_info_score

# KMeans
print("KMeans evaluation: ",mutual_info_score(faces_data.target, labels))

# Spectral cluster
print("Spectral evaluation: ",mutual_info_score(faces_data.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",mutual_info_score(faces_data.target, labels_dbscan))

# AgglomerativeClustering

print("AgglomerativeClustering evaluation: ",mutual_info_score(faces_data.target, labels_Agg))


# In[157]:


# Thực hiện đáng giá theo homogeneity_completeness_v_measure: giá trị trả về trong khoảng 0 >> 1
# - Càng về 1 thì độ khớp của True labels và cluster labels càng cao.

# KMeans
print("KMeans evaluation: ",homogeneity_completeness_v_measure(faces_data.target, labels))

# Spectral cluster
print("Spectral evaluation: ",homogeneity_completeness_v_measure(faces_data.target, labels_spectral))

# DBSCAN
print("DBSCAN evaluation: ",homogeneity_completeness_v_measure(faces_data.target, labels_dbscan))

# AgglomerativeClustering
print("AgglomerativeClustering evaluation: ",homogeneity_completeness_v_measure(faces_data.target, labels_Agg))


# <a id='3.9'></a>
# ### 3.9 Nhận xét
# -  Với data là Faces và sử dụng LBP feature:
#     - Sau khi xem xét evaluation: cho ra kết quả không khả quan.

# <a id='4'></a>
# # Bài 4: Rút trích feature trên dataset tự chọn

# In[158]:


- 4.1 Load dataset: car dataset
- 4.2 Rút trích feature: chọn HoG feature


# <a id='4.1'></a>
# ### 4.1 Load car dataset

# In[178]:


# import thư viện
import glob
from scipy import misc
import imageio
from skimage.feature import hog
from skimage import data, color, exposure

# Load car dataset
# Car dataset gồm 1614 ảnh các góc chụp của nhiều xe 4 bánh
carData = glob.glob('carDataset/*.jpg')
carData += (glob.glob('carDataset/*.png'))
print(len(carData))


# In[179]:


# Show một vài ảnh của data
Figure = plt.figure(figsize=(15,5))

for i in range(995,1005):
    fi = Figure.add_subplot(2, 5, 1+i-995)
    
    # display image
    fi.imshow(imageio.imread(carData[i]), cmap=plt.cm.binary)
    
    plt.axis('off')
    
plt.show()


# <a id='4.2'></a>
# ### 4.2 Rút trích HoG feature

# In[180]:


# Rút trích HoG feature
featureHoG = []

for i in range(1614):
    image = imageio.imread(carData[i])
    image = color.rgb2gray(image)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    featureHoG.append(fd)


# In[181]:


featureHoG = np.array(featureHoG)
print(featureHoG.shape)


# In[182]:


# Áp dụng KMeans trên car dataset
# Với dữ liệu đầu vào là HoG feature
# Chia thành 2 nhóm: nhóm 1 là nhóm có xe, nhóm 2 là nhóm có người
model = KMeans(n_clusters=2)
labels = model.fit_predict(featureHoG)
trueLabels = [0 if i > 1000 else 1 for i in range(1614)]


# In[183]:


# Thống kê kết quả
df=pd.DataFrame({'labels':labels,'Truth labels':trueLabels})
ct = pd.crosstab(df['labels'],df['Truth labels'])
print(ct)


# In[196]:


# Kiểm tra kết quả
Fi = plt.figure(figsize=(15,5))

for n in range(995,1005):
    f = Fi.add_subplot(2, 5, 1+n-995)
    
    # display image
    f.imshow(imageio.imread(carData[n]))
    f.set_title(str(labels[n]))
    plt.axis('off')
plt.show()


# In[190]:


# KMeans evaluation
print("KMeans evaluation: ",adjusted_mutual_info_score(trueLabels, labels))

