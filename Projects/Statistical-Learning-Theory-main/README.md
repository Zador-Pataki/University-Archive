# Statistical-Learning-Theory
Coding exercises of the course Statistical Learning Theory

## Coding Exercise 1
### Image denoising
I denoised the image on the left hand side according to the naive metropolis sampling algorithm, where the gibbs distribution depends on the local energy changes of flipping pixels. Additionally, the algorithm was sped up by performing the calculations in a vectorized format, by alternatively updating two independent subgrids in the image.
<img src="https://user-images.githubusercontent.com/55353663/145596928-20ac09f2-190d-42ca-9a65-3c0280f06c34.png" width="700" height="350">

### Traveling salesman problem
To solve the traveling salesman problem for the cities in america plotted on the left (reach all cities by traveling as short a distance as possible), I implemented the simulated annealing algorithm. To improve the baseline approach, where the proposal distribution either does or does not propose an exchange in checkpoints between two consecutive checkpoints along the route, I implemented a proposal distribution which proposed to exchange a checkpoint with one of 4 neighboring checkpoints along the route. The resulting solution to the problem is presented on the right after a limited number of epochs.
<img src="https://user-images.githubusercontent.com/55353663/145596949-c99449e5-f4a7-4ed4-bcab-2870eee67588.png" width="1100" height="300">

## Coding Exercise 2
### Deterministic Annealing for Clustering
Using deterministic annealing for clustering, the global minimum of the K-Means cost funciton can be efficiently determined. On the left hand side, the procedure is presented on a test case: 4 distinct clusters. When the temperature is high, all cluster centers are located at the mean of all data points. As annealing is performed, and critical temperatures are reached, the cluster centers seperate. At temperature 0, the process is frozen and the global minimas are found. On the corressponding bifurcation plot, the distance of the cluster centers are plotted from their sources: from where they split from another cluster center. 

On the right hand side, the process is performed using a real whine dataset. Using deterministic annealing, we clustered the data into to groups. 
![Screenshot 2021-12-10 162444](https://user-images.githubusercontent.com/55353663/145598608-7e6400d4-fe62-45da-9d16-1cdbf3a49722.png)

After the clustering was complete, I created two plots: the distance of points from the each cluster and the PCA transformation of the data. After this, we I assigned colours to the points: red for red whine and white for white whine. It can be seen that the clusters seperated the datae effectively, dispited the fact that no evident structures in the data can be seen which would seperate these two classes.

<img src="https://user-images.githubusercontent.com/55353663/145598624-524a99c5-3707-4e8e-bf8b-bb94c389034a.png" width="900" height="300">

## Coding Exercise 3
### MAP based Histogram Clustering

On the left hand side, a grid of textures are displayed, with corresponding lables displayed in the middle image. I applied a histogram clustering code using a Maximum a Posteriori approach. A sliding window is passed over the image. In each window, a histogram is constructed and assigned to the center pixel according to the number of pixels with intensities in each pre-defined bin. As a result, each pixel has an assigned histogram, which are then clustered. The result is the image on the right.

![Screenshot 2021-12-10 160506](https://user-images.githubusercontent.com/55353663/145596984-61209a9f-0bd1-44a5-be73-cadc04d6fd72.png)

## Coding Exercise 4
### Constant Shift Embedding
Given proximity data, Constant Shift Embedding is a method for restating pairwise clustering problems in vector spaces while preserving the cluster structure. In this project, I worked with an email-Eu-core network graph based dataset. I reformulated the data as a dissimiolarity matrix, where dissimilarities are minimum distances between points along the graph) plottod in the left hand side off the top row below. I applied the Constant Shift Embedding algorithm and embedded the dissimilarity matrix in the euclidean space, which I clustered into four clusters using the K-Means algorithm. I permuted the dissimilarity matrix accaording to the clusters displayed on the right to demonstrate that the embeddings preserved clusture structures in the proximity data.

For visualizing the data in the embedded space, three further images are plotted. The first two are Constant Shift Embeddings in the 2 and 3 dimensional euclidean space, which were then clustered. The third is the Constant Shift Embedding in a high dimensional euclidean space, clustered and then visualized using PCA. The clusters in the third image aligned with target classes in the data although, no evident clusters are perceivable in the low dimensional space. 
![Screenshot 2021-12-10 161147](https://user-images.githubusercontent.com/55353663/145597007-9eb7c6da-9ee2-4307-90a4-5a5aab1f60a3.png)

## Coding Exercise 5
On the left hand side, the dissimilarity matrix data based on evolutionary distances of proteins is rpesented, where the data is permuted and coloured according to the classification classes in the data. I implemented an EM-like pairwise clustering algorithm using mean field approximations to cluster the data, and then evaluated the performance of the algorithm using multiple scikit-learn metrics which compare the clusters to the ground truth labels. Additionally, the EM-like procedure for pairwise clustering can be re-defined through an embedding based procedure: PCE. On the image on the right hand side, the performances of the methods are compared. The Pairwise Clustering algorithm outperformed existing scikit-learn proximity data clustering algorithms. Additionaly, the embedding based algorithm was similar in performance to the original algorithm.

![Screenshot 2021-12-10 161410](https://user-images.githubusercontent.com/55353663/145597071-52cea0d0-3826-4645-8a3a-decb3c03d19a.png)

The embeddings and corresponding clusters after PCA is plotted below.

![download](https://user-images.githubusercontent.com/55353663/145678913-26cc41e9-3c33-457d-9e91-137bcf536f8e.png)
