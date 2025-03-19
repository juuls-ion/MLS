# Week 4: Vector Search + Vector Databases

Acquiring and processing data is the vital first step of all ML operations. In fact, lack of data/poor data management is a major bottleneck to AI development.

Some definitions:

- *x* - a vector/node
- *q* - the query vector (analogous to the value you “search” on a database)
- *D* - the dataset (of vectors)

### Vector Databases

LLMs have 2 key problems:

- Due to long training times (months for top models), current models can lack up-to-date info.
- Users want the LLMs to not know personal info./details, despite that being a huge volume of training data (e.g. social media profiles, profiles built by Google etc.)

We can encode knowledge into Vector Databases, to be used as “external memory” for models.

- When a model is asked a question, it can “vectorise” the question and check the database for context, as well as its own knowledge.

Vector Databases (DBs) have been around for a while, but recent improvements let you now encode unstructured data, e.g. audio/video.

- These databases can now be used for everything except “relations”, as traditional relational databases (e.g. SQL) are still superior for that.
- As such, Vectors DBs are a very useful tool for storing non-relational data, to complement the ore traditional relational DBs.

### Vector Search

A Vector Search lets you categorise objects by “vectorising” them, allowing you to search for objects similar in certain parameters.

Vector search can be formally defined:

- Given a set of vectors and a query vector, and a distance function defined over the vector space, return the k closest vectors to the query vector, w.r.t. the distance function.
    - Distance functions include Euclidean, dot-product, cosine.
    - These can be normalised also (L2). When normalised, the cosine and dot-product functions are the same.

![Screenshot 2025-03-18 at 6.45.48 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_6.45.48_pm.png)

### kNN Algorithm

The k Nearest Neighbours algorithm is a good, basic algorithm to compute vector search.

It is:

- Highly parallelisable.
- Extremely popular, well understood and analysed (in the literature and in practice).
- Inefficient when the number of samples (*n*) or dimensionality (*d*) are high - both tend to be high in modern ML systems.
    - Also suffers from the **curse of dimensionality:** at high dimensions, the “distance” between points becomes less meaningful.

So, if it is inefficient, how can we approximate it?

- Bear in mind, it is not a slow technique. It is a “brute-force” technique, meaning it finds a 100% complete list of neighbours, becuase it checks every single node/edge.

### Approximate Nearest Neighbours

kNN’s linear-time computation is too slow for bigger datasets [  *O(nd + n min(k, log n))*  ].

So, we precompute “index structures” to use these results to improve live queries. This lets us trade some accuracy for big performance gains.

Note the following definitions:

- **Performance:** Number of vectors visited / number of distance comparisions
    - The fewer comparisons needed (i.e. the higher this value is), the faster the method.
- **Accuracy:** How closely the approximate results match the exact nearest neighbours. There are a few of metrics to measure this.
    
    ![Screenshot 2025-03-18 at 6.58.54 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_6.58.54_pm.png)
    
    - **Recall:** Of all true nearest neighbours, how many did we successfully identify?
        - High recall implies you’re returning most of the correct neighbors.
    - **Precision:** Of all the neighbours returned, how many are actually part of the exact top-*k*?
        - High precision implies that most returned neighbors are correct matches.
    - **Recall@K:**  The fraction of queries for which the approximate top-*k* results overlap with the exact top-*k* results (in whole or part).
        - This metric is especially useful when you only need a few correct neighbors in your approximate results and partial overlap is acceptable.

### Index Functions

The more pre-computations performed, the superior the performance. We use these calculations to generate **Index Functions**, which come from Database Management Systems.

- We construct an index to help us quickly find records that match some **predicate** (logical condition) without having to check every single record.
- Importantly, the index is independent of the query - meaning that once it has been built, it can be used (offline) by many different queries that reference any columns/features it contains.
- With vector data, the predicate would be “return all vectors close to the query vector q” etc. A **vector index** is built using a specialised data structure (see below). The **index function** then returns suitable vectors, without having to compute distances between every single vector and the query.

![Screenshot 2025-03-18 at 7.17.08 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_7.17.08_pm.png)

### ANN Index: Inverted File Index (IVF)

The IVF is one of the specialised data structures we can use to construct our index.

The assumption behind IVF is that similar “things” cluster together, regardless of dataset. This lets us skip most vectors during our search, only searching in the cluster likely to contain nearest neighbours.

The main idea behind IVF is as follows:

- Partition your set of vectors, *D*, into a **Voronoi** **Diagram** (a set of cells).

![Screenshot 2025-03-18 at 7.21.57 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_7.21.57_pm.png)

- You choose some **seed vectors** (here *p1*, *p2*, *p3*) and split the vector space so that, for any given vector in the vector space, it belongs to the cell of the seed vector it is closest to.
    - For any given seed vector, its nearest neighbour will always be within its cell.
    - For any given query vector *q*, its nearest neighbour is **likely** to share its cell. So you then run kNN on the vectors in that cell.
- The traditional database analogy is: the **keys** are the seed vectors and the **values** are the vectors assigned to that seed’s cell.

There are some obvious flaws:

- The cells are generated uniformly based on the seed vectors, so areas of extreme density will make querying in certain cells more/less effective.
- If a query lies near a border, you will probably need to search both cells.

Observations:

- Once we know this diagram, we can construct the IVF.
- Assuming uniform cell size for *K* cells, the IVF is *K* times faster than kNN.

### IVF Construction with K-means Clustering

K-means clustering is the optimal technique to construct the Voronoi diagram. This is just a proven result.

![Screenshot 2025-03-18 at 10.31.10 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_10.31.10_pm.png)

K-means clustering (above) is a technique to generate a set of clusters out of a set of vectors.

- Here a cluster is just a particular group of vectors, we dont draw anything.
- By using some calculus, you can find the mathematically optimal way to partition the vectors into clusters, s.t. each vector belongs to one cluster.
    - By optimal, we mean that each vector is the minimal distance from its cluster’s **centroid** (average vector).
- Then, the centroid of each cluster becomes the seed in the Voronoi diagram.

Then to perform vector search, you do as before: find the closest seed to the query vector, then run kNN in that cell.

### Lloyd’s K-means Algorithm

Alas, K-means clustering is NP-complete (no efficient solution). Lloyd’s algorithm is thus an approximation to K-means:

- Randomly partition your vectors into a set of *k* clusters.
- Compute the centroid of each cluster.
- For each vector *x*, find the centroid it is closest to. If it is the centroid of a different cluster, add it to that cluster.
- Once every vector has been iterated over (and potentially re-assigned to a new cluster), re-compute all the clusters.
- Repeat until convergence (none of the vectors are assigned to new clusters).

Details about the algorithm:

- When the distance function used is Euclidean, it always converges to a **local minimum**.
    - It reaches a good solution, not necessarily the best one (the **global minimum**).
    - This termination is unbounded however, and the worst-case complexity is really bad [ *2^(Ω(√n))* ].
    - Also, not guaranteed to converge with a non-Euclidean distance function.
- The algorithm is fairly simple and easy to implement; it is also extremely parallelizable.
- Average time complexity is decent [ *O(nKd)* ].
- Main weakness is that it is extremely susceptible to the choice of initialisation.
    - This however does make it parallelisable, as you can run different starting seeds on different GPUs.
    - There is something called K-means++ that can be used to initialise better, but they don’t go into that.

### ANN Index: Graph-Based Vector Index

This is a different way to construct the index (as opposed to the IVF-Voronoi method).

We represent the entire dataset as a graph, with nearer vectors more likely to be connected by edges:

- Starting from an initial point, you traverse the graph, trying to get as close to the query vector as possible.
    - The nearest neighbours are the nodes closest (according to your distance function) to the query.
- The vector seach problem is therefore reduced to a graph traversal problem.
    - The entire graph may be hugely complex to derive, so we can treat it as unknown. We navigate using a local strategy (only need to know the neighbouring nodes), the **greedy routing strategy**:
        - Start at your current node, measure the distance to the query.
        - Measure the distance from all your neighbours to the query.
        - Pick the closest neighbour as the next node.
        - If current node is the closest, you have converged.
    - The greedy strategy is very efficient [ *O (log n)* ]. It is also guaranteed to work if the graph is **navigable**.

### Delaunay Graphs

Navigability is a relevant property of some graphs:

![Screenshot 2025-03-18 at 11.13.14 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_11.13.14_pm.png)

This is important, because there is a special type of navigable graph called the **Delaunay Graph.**

- The **exact** nearest neighbour can be found using the greedy routing algorithm on a Delaunay Graph of a dataset *D*.
- To construct the Delaunay Graph (DG), first make a Voronoi diagram of *D*, where every single node is a seed (i.e. one cell per vector).
- Then draw edges between all nodes whose cells share an adjacent edge.
    - This is efficient, as for any circle drawn between two vectors connected by an edge, there will be no other vectors in the circle (i.e. edges are sparse).
- Cheap to construct as complexity: total # edges examined (i.e., # steps * average vertex degree).

![Screenshot 2025-03-18 at 11.18.43 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-18_at_11.18.43_pm.png)

The Delaunay Graph (DG) cannot be used for direct kNN queries, as it becomes exponentially more complex to construct in higher dimensions. Furthermore, at high dimensions it approximates a fully connected graph (so kNN over this graph essentially reduces to brute force).

So, we try to approximate the DG using a sparser graph. Specifically, a sparser navigable graph. The sparser the graph is, the fewer edges are examined, the lower the complexity when constructing it.

- Remember, the graph needs to be navigable, so that the greedy routing algorithm will work.
    - There is a sparsity limitation. Any navigable graph will have an average degree of [ *Ω( √n / log n)* ].

### Navigable Small World Graphs

The Delaunay Graph is a type of **Small World Graph (SWG)**. These are networks where most nodes can be reached from any other by a small number of steps, despite the network beung potentially very large.

- The classic analogy is social networks: the idea that anyone is only “six degrees of seperation” from anyone else on Earth, even though most people only know individuals in their social circles (local graph clusters).

![Screenshot 2025-03-19 at 2.46.35 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-19_at_2.46.35_pm.png)

We want to try and build a sparse small world graph to approximate the Delaunay Graph (which is also made of many small, local connections).

- One problem: in SWGs, the number of steps in a path can be very high (since the connections are short-distance), meaning high computational cost.
- The solution: Random rewiring to add a few super-long edges. This has been shown to massively decrease the number of “hops”, especially as the distance to the query increases.
    - The social network analogy is: imagine you live in London. It would take ages to connect yourself to someone living in Glasgow by snaking your way up through the country using short connections. If however you know someone in Edinburgh, it becomes way quicker.

![Screenshot 2025-03-19 at 2.49.49 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-19_at_2.49.49_pm.png)

Remember however, one of our key constraints is **navigability**. How do we ensure these SWGs are navigable (to make Navigable Small World Graphs - NSWGs).

- Remember, the fully-connected graph (all nodes connected to all other nodes) is obviously navigable. The Dalaunay Graph is a subset of the fully-connected graph, and is also Navigable. We are trying to approximate the Delaunay Graph (i.e. even smaller), while retaining navigability.

Making these NSWGs is complex and can only be approximately done using Heuristics:

- Start with your set of vectors, and randomly add one to the graph.
- Add another random vector, and connect it to its k-nearest neighbours that are already in the graph.
    - The idea is that the early vectors you add will be randomly spread, and provide the long-range connections. Later nodes will add all the short connections (as the more nodes already in the graph, the closer each k-nearest neighbour will be).

![Screenshot 2025-03-19 at 2.58.01 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-19_at_2.58.01_pm.png)

The resulting graph is (approximately) navigable and small-world. These “generated” graphs tend to be less efficient to build and query that “natural” NSWGs (like the Delaunay Graph).

- Query complexity estimated at *O ( log ^2 n)*, build complexity at *O (n log ^2 n)*.

### HNSW (Heirarchical Navigable Small World)

The reason the “generated” NSWGs are less efficient is because during generation, some vertices develop very large **degrees** (i.e. connected to many other vectors), becoming “hub nodes”.

- Remember the greedy algorithm “scans over” all the neighbours of a node, so this directly decreases performance.

So to optimise, we build the graph up in layers!

- The bottom layer is the original “generated” NSWG.
- One layer up, you sample only some of the vertices, regenerating the graph (using fewer nodes).
- This iterates for multiple layers, until the top layer has very few, distantly connected nodes.
- There is also a heuristic that prevents dense “hub nodes”: We do not add any edge that would become the longest edge of a triangle (because this is an unnecessary edge, the connection between the two nodes must already exist).

To query the graph, we start at the entry node (usually arbitrarily chosen) on the top layer. 

- We perform greedy search on the top layer, until we reach the nearest neighbour.
- We use this node as the entry node for the next layer, again performing greedy search.
- Repeat until you hit the bottom layer (original NSWG). The nearest neighbours found at this layer are the true nearest neighbours!

![Screenshot 2025-03-19 at 3.15.29 pm.png](Week%204%20Vector%20Search%20+%20Vector%20Databases%201ba65b33432380508fb1e3d3554aa2ea/Screenshot_2025-03-19_at_3.15.29_pm.png)