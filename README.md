# Using-Eigenvectors-for-Graph-Partition

In this project, I focus on using eigenvectors and K-means clustering method to partition a set of vertices V into k groups 𝑉1, 𝑉2..., 𝑉𝑘 in an undirected graph G = (V, E). 

## Getting Started

These introduction will get you a copy of the project up and running on your local machine purposes. See deployment for notes on how to deploy this project on a live system.

### Algorithms

Input graph G, number k

     		1. Form graph adjacency matrix A
     		2. Form diagonal matrix D
     		3. Form normalized Laplacian matrix *L' = I -  $D^{-1/2}AD^{-1/2}$*
     		4. Compute the first k eigenvectors $u_1,...,u_k$ of L'​
     		5. Form matrix $U ∈ R^{n×k}$ with columns $u_1,...,u_k$
     		6. Normalize $U$ so that rows have norm 1
     		7. Consider the i-th row of $U$ as point $y_i ∈ R^k,i=1,...,n$ 
     		8. Cluster the points {$y_i$}$i=1,...,n$ into clusters $C_1,...C_k$, using k-means clustering

### Prerequisites

* Python3

* numpy

* sklearn

* scipy

### Installing

First install [python3](https://www.python.org/downloads/), then using pip command to download required packages that demonstrate above.

For Example:   

```python
pip3 install numpy
```

### Running the project

a. In command line, type this command in the ***code*** directory:

```
python3 graph_partitioning.py 
```

b. In pycharm:

Just open this project and configure the running environment as we said above. Then run the ```graph_partitioning.py ``` python file.



You can change the test file by altering the input txt filename in the bottom of ```graph_partitioning.py ```:

```python
if __name__ == '__main__':
    labels = partitioning_1('Oregon-1.txt')
    for i in labels:
        print(i)
        print(len(i))
```

Change **Orgen-1.txt** to other filenames in ***data*** directory.  

### Authors

Xiaobo Wu, Aalto University

## License

This project is licensed under the MIT License - see [this page](https://opensource.org/licenses/MIT) for details.

## Acknowledgments

* Konstantin Andreev, Harald Racke. (2006). Balanced Graph Partitioning. Theory

  of Computing Systems.

* Bojan Mohar. The Laplacian Spectrum of Graphs. University of Ljubljna.

* https://www.geeksforgeeks.org/sparse-matrix-representation

* https://en.wikipedia.org/wiki/Laplacian_matrix

* https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

* https://en.wikipedia.org/wiki/Graph_partition

  