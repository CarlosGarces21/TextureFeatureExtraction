import numpy as np
import cv2
import pandas as pd

img = cv2.imread('TrainingImages/2410.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save original image pixels into a data frame.
img2 = img.reshape(-1) #convert image to 1D
df = pd.DataFrame() #creating dataframe
df['Original Image'] = img2 #image in 1D in the data set

# Generate Gabor features
num = 1  # To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(4):  # Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  # Sigma variation
        for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
            for gamma in (0.05, 0.3, 0.5):  # Gamma values variation

                gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                #                print(gabor_label)
                ksize = 12
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                # Now filter the image and add values to a new column
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)

                cv2.imwrite('GaborFilterBank/' + gabor_label + '.jpg', filtered_img.reshape(img.shape))

                df[gabor_label] = filtered_img  # Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  # Increment for gabor column label

########################################
# Gerate OTHER FEATURES and add them to the data frame

# CANNY EDGE
edges = cv2.Canny(img, 100, 200)  # Image, min and max values
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1  # Add column to original dataframe
cv2.imwrite('FilterBank/edgeCanny.jpg', edges)

from skimage.filters import roberts, sobel, scharr, prewitt

# ROBERTS EDGE
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1
cv2.imwrite('FilterBank/edgeRoberts.jpg', edge_roberts)

# SOBEL
edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1
cv2.imwrite('FilterBank/edgeSobel.jpg', edge_sobel)

# SCHARR
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1
cv2.imwrite('FilterBank/edgeScharr.jpg', edge_scharr)

# PREWITT
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1
cv2.imwrite('FilterBank/edgePrewitt.jpg', edge_prewitt)

from scipy import ndimage as nd

# # GAUSSIAN with sigma2
# gaussian_img = nd.gaussian_filter(img, sigma=2)
# gaussian_img1 = gaussian_img.reshape(-1)
# df['Gaussian s2'] = gaussian_img1
# cv2.imwrite('FilterBank/gaussian1.jpg', gaussian_img)
#
# # GAUSSIAN with sigma=3
# gaussian_img2 = nd.gaussian_filter(img, sigma=3)
# gaussian_img3 = gaussian_img2.reshape(-1)
# df['Gaussian s3'] = gaussian_img3
# cv2.imwrite('FilterBank/gaussian2.jpg', gaussian_img2)
#
# # GAUSSIAN with sigma=4
# gaussian_img4 = nd.gaussian_filter(img, sigma=4)
# gaussian_img5 = gaussian_img4.reshape(-1)
# df['Gaussian s4'] = gaussian_img5
# cv2.imwrite('FilterBank/gaussian1.jpg', gaussian_img4)

# GAUSSIAN with sigma=5
gaussian_img6 = nd.gaussian_filter(img, sigma=5)
gaussian_img7 = gaussian_img6.reshape(-1)
df['Gaussian s5'] = gaussian_img7
cv2.imwrite('FilterBank/gaussian1.jpg', gaussian_img6)

# GAUSSIAN with sigma=7
gaussian_img8 = nd.gaussian_filter(img, sigma=7)
gaussian_img9 = gaussian_img8.reshape(-1)
df['Gaussian s7'] = gaussian_img9
cv2.imwrite('FilterBank/gaussian1.jpg', gaussian_img8)

# MEDIAN with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

# # VARIANCE with size=3
# variance_img = nd.generic_filter(img, np.var, size=3)
# variance_img1 = variance_img.reshape(-1)
# df['Variance s3'] = variance_img1  # Add column to original dataframe

# print(df.head())
######################################

# Now, add a column in the data frame for the Labels
# For this, we need to import the labeled image
labeled_img = cv2.imread('TrainingImages/2410_spot_background.ome.tiff')
# Remember that you can load an image with partial labels
# But, drop the rows with unlabeled data

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1

print(df.head())

# df.to_csv("Gabor.csv")

# maybe needed
original_img_data = df.drop(labels = ["Labels"], axis=1) # use for prediction
# df.to_csv("LabelsperPixel.csv")
df = df[df.Labels != 0]

#########################################################

# Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

# Define the independent variables
X = df.drop(labels=["Labels"], axis=1)

# Split data into train and test to verify accuracy after fitting the model.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

# Import the model we are using
# RandomForestRegressor is for regression type of problems.
# For classification we use RandomForestClassifier.
# Both yield similar results except for regressor the result is float
# and for classifier it is an integer.

from sklearn.ensemble import RandomForestClassifier

# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on training data
model.fit(X_train, y_train)

# verify number of trees used. If not defined above.
# print('Number of Trees used : ', model.n_estimators)

# STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
# AND CALCULATE THE ACCURACY SCORE
# First test prediction on the training data itself. SHould be good.
prediction_test_train = model.predict(X_train)

# Test prediction on testing data.
prediction_test = model.predict(X_test)

# .predict just takes the .predict_proba output and changes everything
# to 0 below a certain threshold (usually 0.5) respectively to 1 above that threshold.
# In this example we have 4 labels, so the probabilities will for each label stored separately.
# prediction_prob_test = model.predict_proba(X_test)

# Let us check the accuracy on test data
from sklearn import metrics

# Print the prediction accuracy
# First check the accuracy on training data. This will be higher than test data prediction accuracy.
print("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
# Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
# This part commented out for SVM testing. Uncomment for random forest.
# One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
# importances = list(model.feature_importances_)
# Let us print them into a nice format.
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

#SVM model
#SVM
# Train the Linear SVM to compare against Random Forest
from sklearn.svm import LinearSVC
model_SVM = LinearSVC(max_iter=500)
model_SVM.fit(X_train, y_train)

#Test prediction on testing data.
prediction_RF = model.predict(X_test)
prediction_SVM = model_SVM.predict(X_test)
#prediction_LR = model_LR.predict(X_test)

# Metrics:
from sklearn import metrics

# Print the prediction accuracy
# Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print("Accuracy using Random Forest= ", metrics.accuracy_score(y_test, prediction_RF))
print("Accuracy using SVM = ", metrics.accuracy_score(y_test, prediction_SVM))
# print ("Accuracy using LR = ", metrics.accuracy_score(y_test, prediction_LR))


from yellowbrick.classifier import ROCAUC
print("Classes in the image are: ", np.unique(Y))

# # ROC curve for RF
roc_auc = ROCAUC(model, classes=[0, 1])  # Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()

# # ROC curve for SVM
# roc_auc = ROCAUC(model_SVM, classes=[1, 2])  # Create object
# roc_auc.fit(X_train, y_train)
# roc_auc.score(X_test, y_test)
# roc_auc.show()

# MAKE THE PREDICTION
import pickle

# Save the trained model as pickle string to disk for future use
filename = "MLmodels/MLmodel_Spot4"
pickle.dump(model, open(filename, 'wb'))

# To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(original_img_data)

segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt

plt.imshow(segmented, cmap='jet')
plt.imsave('TestResultSpot4.png', segmented, cmap='jet')