# cse6242-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [CSE6242-Homework 4 Solved](https://mantutor.com/product/cse6242-homework-4-solved-2/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;92084&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE6242-Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
<h1><a name="_Toc20753"></a>Homework Overview</h1>
Data analytics and machine learning both revolve around using computational models to capture relationships between variables and outcomes. In this assignment, you will code and fit a range of well-known models from scratch and learn to use a popular Python library for machine learning.

&nbsp;

In Q1, you will implement the famous PageRank algorithm from scratch. PageRank can be thought of as a model for a system in which a person is surfing the web by choosing uniformly at random a link to click on at each successive webpage they visit. Assuming this is how we surf the web, what is the probability that we are on a particular webpage at any given moment? The PageRank algorithm assigns values to each webpage according to this probability distribution.

&nbsp;

In Q2, you will implement Random Forests, a very common and widely successful classification model, from scratch. Random Forest classifiers also describe probability distributions—the conditional probability of a sample belonging to a particular class given some or all of its features.

&nbsp;

Finally, in Q3, you will use the Python scikit-learn library to specify and fit a variety of supervised and unsupervised machine learning models.

<h1><a name="_Toc20754"></a>Q1&nbsp; Implementation of Page Rank Algorithm</h1>
<strong>Note: You must use Python 3.7.x for this question. </strong>

&nbsp;

<table width="698">
<tbody>
<tr>
<td width="137">Technology</td>
<td width="561">PageRank Algorithm

Graph

Python 3.7.x
</td>
</tr>
<tr>
<td width="137">Allowed Libraries</td>
<td width="561">NA</td>
</tr>
<tr>
<td width="137"></td>
<td width="561"></td>
</tr>
<tr>
<td width="137"></td>
<td width="561"></td>
</tr>
</tbody>
</table>
&nbsp;

In this question, you will implement the PageRank algorithm in Python for a large graph network dataset.

&nbsp;

The PageRank algorithm was first proposed to rank web pages in search results. The basic assumption is that more “important” web pages are referenced more often by other pages and thus are ranked higher. The algorithm works by considering the number and “importance” of links pointing to a page, to estimate how important that page is. PageRank outputs a probability distribution over all web pages, representing the likelihood that a person randomly surfing the web (randomly clicking on links) would arrive at those pages.

&nbsp;

As mentioned in the lectures, the PageRank values are the entries in the dominant eigenvector of the modified adjacency matrix in which each column’s values adds up to 1 (i.e., “column normalized”), and this eigenvector can be calculated by the power iteration method that you will implement in this question, which iterates through the graph’s edges multiple times to update the nodes’ PageRank values (“pr_values” in pagerank.py) in each iteration.

&nbsp;

For each iteration, the PageRank computation for each node in the network graph is

&nbsp;

𝑃𝑅<sub>𝑡</sub>(𝑣<sub>𝑖</sub>)

𝑃𝑅<sub>𝑡+1</sub>(𝑣<sub>𝑗</sub>) = (1 − 𝑑) × 𝑃𝑑(𝑣<sub>𝑗</sub>) + 𝑑 × ∑𝑜𝑢𝑡&nbsp;𝑑𝑒𝑔𝑟𝑒𝑒(𝑣<sub>𝑖</sub>)

𝑣<sub>𝑖 </sub>for each edge (𝑣<sub>𝑖</sub>, 𝑣<sub>𝑗</sub>) from 𝑣<sub>𝑖</sub> to 𝑣<sub>𝑗</sub>, where

<ul>
<li>𝑣<sub>𝑗</sub> is node 𝑗</li>
<li>𝑣<sub>𝑖</sub> is node 𝑖 that points to node 𝑗</li>
<li>𝑜𝑢𝑡 𝑑𝑒𝑔𝑟𝑒𝑒(𝑣<sub>𝑖</sub>) is the number of links going out of node 𝑣<sub>𝑖</sub></li>
<li>𝑃𝑅<sub>𝑡+1</sub>(𝑣<sub>𝑗</sub>) is the pagerank value of node 𝑗 at iteration 𝑡 + 1</li>
<li>𝑃𝑅<sub>𝑡</sub>(𝑣<sub>𝑖</sub>) is the pagerank value of node 𝑖 at iteration 𝑡</li>
<li>𝑑 is the damping factor; set it to the common value of 85 that the surfer would continue to follow links</li>
<li>𝑃𝑑(𝑣<sub>𝑗</sub>) is the probability of random jump that can be personalized based on use cases</li>
</ul>
<h2><a name="_Toc20755"></a>Tasks</h2>
You will be using the “network.tsv” graph network dataset in the hw4-skeleton/Q1 folder, which contains about 1 million nodes and 3 million edges. Each row in that file represents a directed edge in the graph.&nbsp; The edge’s source node id is stored in the first column of the file, and the target node id is stored in the second column. <strong>Note:</strong> your code must <strong>NOT</strong> make any assumptions about the relative magnitude between the node ids of an edge. For example, suppose we find that the source node id is smaller than the target node id for most edges in a graph, we must <strong>NOT</strong> assume that this is always the case for all graphs (i.e., in other graphs, a source node id can be larger than a target node id).

&nbsp;

You will complete the code in submission.py&nbsp; (guidelines also provided in the file).

<ul>
<li>Step 1: in calculate_node_degree(), calculate and store each node’s out-degree and the graph’s maximum node id.
<ul>
<li>A node’s out-degree is its number of outgoing edges. Store the out-degree in class variable “node_degree”.</li>
<li>max_node_id refers to the highest node id in the graph. For example, suppose a graph contains the two edges (1,4) and (2,3), in the format of (source,target), the max_node_id here is 4. Store the maximum node id to class variable ● Step 2: implement run_pagerank()</li>
<li>For simplified PageRank algorithm, where<em> Pd( v<sub>j </sub>)</em> = 1/(max_node_id + 1) is provided as node_weights in the script and you will submit the output for 10 and 25 iteration runs for a damping factor of 0.85. To verify, we are providing the sample output of 5 iterations for a simplified PageRank (simplified_pagerank_iter5_sample.txt<strong>)</strong>. For personalized PageRank, the <em>Pd( )</em> vector will be assigned values based on your 9-digit GTID (e.g., 987654321) and you will submit the output for 10 and 25 iteration runs for a damping factor of 0.85.</li>
</ul>
</li>
<li>The beginning of the main function in py describes how to run the algorithm and generate output files. <strong>Note: </strong>When comparing your output for simplified_pagerank for 5 iterations with the given sample output, the absolute difference must be less than 5%, i.e., Absolute((SampleOutput – YourOutput)/SampleOutput) must be less than 0.05.</li>
</ul>
<h1><a name="_Toc20756"></a>Q2 Random Forest Classifier</h1>
<table width="700">
<tbody>
<tr>
<td width="138">Technology</td>
<td width="563">Python 3.7.x</td>
</tr>
<tr>
<td width="138">Allowed Libraries</td>
<td width="563">Do not modify the import statements; everything you need to complete this question has been imported for you. You may not use other libraries for this assignment.</td>
</tr>
<tr>
<td width="138"></td>
<td width="563"></td>
</tr>
<tr>
<td width="138"></td>
<td width="563"></td>
</tr>
</tbody>
</table>
<h2><a name="_Toc20757"></a>Q2.1 – Random Forest Setup</h2>
<strong>&nbsp;</strong>

<strong>Note: You must use Python 3.7.x for this question. </strong>

&nbsp;

You will implement a random forest classifier in Python via a <a href="https://jupyter.readthedocs.io/en/latest/install.html">Jupyter notebook</a><a href="https://jupyter.readthedocs.io/en/latest/install.html">.</a> The performance of the classifier will be evaluated via the out-of-bag (OOB) error estimate using the provided dataset pimaindians-diabetes.csv, a comma-separated (csv) file in the Q2 folder. The dataset was derived from the <a href="https://data.world/uci/pima-indians-diabetes">National Institute of Diabetes and Digestive and </a><a href="https://data.world/uci/pima-indians-diabetes">Kidney Diseases</a><a href="https://data.world/uci/pima-indians-diabetes">.</a><strong> You must not modify the dataset.</strong> Each row describes one person (a data point, or data record) using 9 columns. The first 8 are attributes. The 9th is the label, and you must <strong>NOT</strong> treat it as an attribute. You will perform binary classification on the dataset to determine if a person has diabetes.

&nbsp;

<strong>Important:</strong>

<ol>
<li>Remove all “testing” code that renders output, or Gradescope will crash. For instance, any additional print, display, and show statements used for debugging must be removed.</li>
</ol>
&nbsp;

<ol start="2">
<li>You may only use the modules and libraries provided at the top of the notebook file included in the skeleton for Q2 and modules from the Python Standard Library. Python wrappers (or modules) must <strong>NOT</strong> be used for this assignment. Pandas must <strong>NOT</strong> be used — while we understand that they are useful libraries to learn, completing this question is not critically dependent on their functionality. In addition, to make grading more manageable and to enable our TAs to provide better, more consistent support to our students, we have decided to restrict the libraries accordingly.</li>
</ol>
<h3>Essential Reading</h3>
<strong>Decision Trees.</strong> To complete this question, you will develop a good understanding of how decision trees work. We recommend that you review the lecture on the decision tree. Specifically, review how to construct decision trees using <em>Entropy </em>and<em> Information Gain</em> to select the splitting attribute and split point for the selected attribute. These <a href="http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf">slides from CMU</a> (also mentioned in the lecture) provide an excellent example of how to construct a decision tree using <em>Entropy</em> and <em>Information Gain</em>. <strong>Note: </strong>there is a typo on page 10, containing the Entropy equation; ignore one negative sign (only one negative sign is needed).

&nbsp;

<strong>Random Forests.</strong> To refresh your memory about random forests,&nbsp; see Chapter 15 in the <a href="https://web.stanford.edu/~hastie/Papers/ESLII.pdf">Elements of</a> <a href="https://web.stanford.edu/~hastie/Papers/ESLII.pdf">Statistical Learning</a> book and the lecture on random forests. Here is a <a href="http://blog.echen.me/2011/03/14/laymans-introduction-to-random-forests/">blog post</a> that introduces random forests in a fun way, in layman’s terms.

<strong>&nbsp;</strong>

<strong>Out-of-Bag Error Estimate. </strong>In random forests, it is not necessary to perform explicit cross-validation or use a separate test set for performance evaluation. Out-of-bag (OOB) error estimate has shown to be reasonably accurate and unbiased. Below, we summarize the key points about OOB in the <a href="https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr">original article</a> <a href="https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr">by Breiman and Cutler</a><a href="https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr">.</a>

&nbsp;

Each tree in the forest is constructed using a different bootstrap sample from the original data. Each bootstrap sample is constructed by randomly sampling from the original dataset <strong>with replacement </strong>(usually, a bootstrap sample has the <a href="http://stats.stackexchange.com/questions/24330/is-there-a-formula-or-rule-for-determining-the-correct-sampsize-for-a-randomfore">same size</a> as the original dataset). Statistically, about one-third of the data records (or data points) are left out of the bootstrap sample and not used in the construction of the <em>kth</em> tree. For each data record that is not used in the construction of the <em>kth</em> tree, it can be classified by the <em>kth</em> tree. As a result, each record will have a “test set” classification by the subset of trees that treat the record as an out-of-bag sample. The majority vote for that record will be its predicted class. The proportion of times that a record’s predicted class is different from the true class, averaged over all such records, is the OOB error estimate.

&nbsp;

While splitting a tree node, make sure to randomly select a subset of attributes (e.g., square root of the number of attributes) and pick the best splitting attribute (and splitting point of that attribute) among these subsets of attributes. This randomization is the main difference between random forest and bagging decision trees.

&nbsp;

<h4>Starter Code</h4>
We have prepared some Python starter code to help you load the data and evaluate your model. The starter file name is Q2.ipynb has three classes:

<ul>
<li>Utililty: contains utility functions that help you build a decision tree</li>
<li>DecisionTree: a decision tree class that you will use to build your random forest ● RandomForest: a random forest class</li>
</ul>
&nbsp;

<h4>What you will implement</h4>
Below, we have summarized what you will implement to solve this question. Note that you must use <strong>information gain</strong> to perform the splitting in the decision tree. The starter code has detailed comments on how to implement each function.

<ol>
<li>Utililty class: implement the functions to compute entropy, information gain, perform splitting, and find the best variable (attribute) and split-point. You can add additional methods for convenience. Note: Do not round the output or any of your functions.</li>
<li>DecisionTree class: implement the learn() method to build your decision tree using the utility functions above.</li>
<li>DecisionTree class: implement the classify() method to predict the label of a test record using your decision tree.</li>
<li>RandomForest class: implement the methods _bootstrapping(), fitting(), voting() and user().</li>
<li>get_random_seed(), get_forest_size(): implement the functions to return a random seed and forest size (number of decision trees) for your implementation.</li>
</ol>
&nbsp;

<strong>&nbsp;</strong>

<strong>Important</strong>:

<ol>
<li>You <strong>must</strong> achieve a minimum accuracy of <strong>70%</strong> for the random forest.</li>
<li>Your code must take <strong>no more than 5 minutes</strong> to execute (which is a very long time, given the low program complexity). Otherwise, it may time out on Gradescope. Code that takes longer than 5 minutes to run likely means you need to correct inefficiencies (or incorrect logic) in your program. We suggest that you check the hyperparameter choices (e.g., tree depth, number of trees) and code logic when figuring out how to reduce runtime.</li>
<li>The run()function is provided to test your random forest implementation; do <strong>NOT</strong> modify this function.</li>
</ol>
&nbsp;

As you solve this question, consider the following design choices. Some may be more straightforward to determine, while some maybe not (hint: study lecture materials and essential reading above). For example:

<ul>
<li>Which attributes to use when building a tree?</li>
<li>How to determine the split point for an attribute?</li>
<li>How many trees should the forest contain?</li>
<li>You may implement your decision tree using a data structure of your choice (e.g., dictionary, list, class member variables). However, your implementation must still work within the DecisionTree Class Structure we have provided.</li>
<li>Your decision tree will be initialized using DecisionTree(max_depth=10), in the RandomForest class in the jupyter notebook.</li>
<li>When do you stop splitting leaf nodes?</li>
<li>The depth found in the learn function is the depth of the current node/tree. You may want a check within the learn function that looks at the current depth and returns if the depth is greater than or equal to the max depth specified. Otherwise it is possible that you continually split on nodes and create a messy tree. The max_depth parameter should be used as a stopping condition for when your tree should stop growing. Your decision tree will be instantiated with a depth of 0 (input to the learn() function in the jupyter notebook). To comply with this, make sure you implement the decision tree such that the root node starts at a depth of 0 and is built with increasing depth.</li>
</ul>
&nbsp;

Note that, as mentioned in the lecture, there are other approaches to implement random forests. For example, instead of information gain, other popular choices include the Gini index, random attribute selection (e.g., <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.2940&amp;rep=rep1&amp;type=pdf">PERT </a><a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.2940&amp;rep=rep1&amp;type=pdf">– </a><a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.2940&amp;rep=rep1&amp;type=pdf">Perfect Random Tree Ensembles</a><a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.2940&amp;rep=rep1&amp;type=pdf">)</a>. We decided to ask everyone to use an information gain based approach in this question (instead of leaving it open-ended), to help standardize students’ solutions to help accelerate our grading efforts.

&nbsp;

<h2><a name="_Toc20758"></a>Q2.2 – Random Forest Reflection</h2>
On Gradescope, answer the following question:

What is the main reason to use a random forest versus a decision tree?

&nbsp;

<h1><a name="_Toc20759"></a>Q3 Using Scikit-Learn</h1>
<table width="700">
<tbody>
<tr>
<td width="138">Technology</td>
<td width="563">Python 3.7.x

Scikit-Learn 0.22
</td>
</tr>
<tr>
<td width="138">Allowed Libraries</td>
<td width="563">Do not modify the import statements; everything you need to complete this question has been imported for you. You may not use other libraries for this assignment.</td>
</tr>
<tr>
<td width="138"></td>
<td width="563"></td>
</tr>
<tr>
<td width="138"></td>
<td width="563"></td>
</tr>
</tbody>
</table>
&nbsp;

<strong>&nbsp;</strong>

<a href="http://scikit-learn.org/">Scikit</a><a href="http://scikit-learn.org/">–</a><a href="http://scikit-learn.org/">learn</a> is a popular Python library for machine learning. You will use it to train some classifiers to predict diabetes in the Pima Indian tribe. The dataset is provided in the Q3 folder as pima-indiansdiabetes.csv.

&nbsp;

For this problem, you will be utilizing a <a href="https://jupyter.readthedocs.io/en/latest/install.html">Jupyter</a> <a href="https://jupyter.readthedocs.io/en/latest/install.html">notebook</a><a href="https://jupyter.readthedocs.io/en/latest/install.html">.</a>

&nbsp;

<strong>Important</strong>:

<ol>
<li>Remove all “testing” code that renders output, or Gradescope will crash. For instance, any additional print, display, and show statements used for debugging must be removed.</li>
<li>Use the default values while calling functions unless specific values are given.</li>
<li>Do not round off the results except the results obtained for Linear Regression Classifier.</li>
</ol>
<h2><a name="_Toc20760"></a>Q3.1 – Data Import</h2>
In this step, you will import the pima-indians-diabetes dataset and allocate the data to two separate arrays. After importing the data set, you will split the data into a training and test set using the scikit-learn function <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">.</a> You will use scikit-learns built-in machine learning algorithms to predict the accuracy of training and test set separately. Refer to the hyperlinks provided below for each algorithm for more details, such as the concepts behind these classifiers and how to implement them.

<h2><a name="_Toc20761"></a>Q3.2 – Linear Regression Classifier</h2>
Q3.2.1 – Classification

Train the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression</a> classifier on the dataset. You will provide the accuracy for both the test and train sets. Make sure that you round your predictions to a binary value of 0 or 1. See the Jupyter notebook for more information. Linear regression is most commonly used to solve regression problems. The exercise here demonstrates the possibility of using linear regression for classification (even though it may not be the optimal model choice).

<h2><a name="_Toc20762"></a>Q3.3 – Random Forest Classifier</h2>
Q3.3.1 – Classification

Train the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest</a> classifier on the dataset. You will provide the accuracy for both the test and train sets. Do not round your prediction.

&nbsp;

Q3.3.2 – Feature Importance

You have performed a simple classification task using the random forest algorithm. You have also implemented the algorithm in Q2 above. The concept of entropy gain can also be used to evaluate the importance of a feature. You will determine the feature importance evaluated by the random forest classifier in this section. Sort the features in descending order of feature importance score, and print the sorted features’ numbers.

&nbsp;

<strong>Hint:</strong> There is a function available in sklearn to achieve this. Also, take a look at argsort() function in Python numpy. argsort()returns the indices of the elements in ascending order. You

will use the random forest classifier that you trained initially in Q3.3.1, without any kind of hyperparameter-tuning, for reporting these features.

&nbsp;

Q3.3.3 – Hyper-Parameter Tuning

Tune your random forest hyper-parameters to obtain the highest accuracy possible on the dataset. Finally, train the model on the dataset using the tuned hyper-parameters. Tune the hyperparameters specified below, using the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> function in Scikit library:

&nbsp;

<strong>&nbsp;‘n_estimators’: [4, 16, 256], ’max_depth’: [2, 8, 16]</strong>

<h2><a name="_Toc20763"></a>Q3.4 – Support Vector Machine</h2>
Q3.4.1 – Preprocessing

For SVM, we will standardize attributes (features) in the dataset using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">,</a> before training the model.

&nbsp;

<strong>Note:</strong> for StandardScaler,

<ul>
<li>Transform both x_train and x_test to obtain the standardized versions of both.</li>
<li>Review the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler documentation</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">,</a> which provides details about standardization and how to implement it.</li>
</ul>
&nbsp;

Q3.4.2 – Classification

Train the <a href="https://scikit-learn.org/stable/modules/svm.html">Support Vector Machine</a> classifier on the dataset (the link points to SVC, a particular implementation of SVM by Scikit). You will provide the accuracy on both the test and train sets.

&nbsp;

Q3.4.3. –&nbsp; Hyper-Parameter Tuning

Tune your SVM model to obtain the highest accuracy possible on the dataset. For SVM, tune the model on the standardized train dataset and evaluate the tuned model with the test dataset. Tune the hyperparameters specified below, using the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> function in Scikit library:

&nbsp;

<strong>‘kernel’:(‘linear’, ‘rbf’), ‘C’:[0.01, 0.1, 1.0]</strong>

&nbsp;

<strong>Note: </strong>If GridSearchCV takes a long time to run for SVM, make sure you standardize your data beforehand using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">.</a>

&nbsp;

Q3.4.4. – Cross-Validation Results

Let’s practice obtaining the results of cross-validation for the SVM model. Report the rank test score and mean testing score for the best combination of hyper-parameter values that you obtained. The GridSearchCV class holds a cv_results_ dictionary that helps you report these metrics easily.

&nbsp;

<h2><a name="_Toc20764"></a>Q3.5 – Principal Component Analysis</h2>
Performing <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">Principal Component Analysis</a> based dimensionality reduction is a common task in many data analysis tasks, and it involves projecting the data to a lower-dimensional space using Singular Value Decomposition. Refer to the examples given <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">here</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">;</a> set parameters n_component to 8 and svd_solver to full. See the sample outputs below.

&nbsp;

<ol>
<li>Percentage of variance explained by each of the selected components. Sample Output:</li>
</ol>
&nbsp;

[6.51153033e-01 5.21914311e-02 2.11562330e-02 5.15967655e-03

6.23717966e-03 4.43578490e-04 9.77570944e-05 7.87968645e-06]

&nbsp;

<ol start="2">
<li>The singular values corresponding to each of the selected components. Sample Output:</li>
</ol>
<strong>&nbsp;</strong>

[5673.123456&nbsp; 4532.123456&nbsp;&nbsp; 4321.68022725&nbsp; 1500.47665361

1250.123456&nbsp;&nbsp; 750.123456&nbsp;&nbsp;&nbsp; 100.123456&nbsp;&nbsp;&nbsp; 30.123456]

&nbsp;

<strong>Use the Jupyter notebook skeleton file called Q3.ipynb to write and execute your code. </strong>

&nbsp;

As a reminder, the general flow of your machine learning code will look like:

<ol>
<li>Load dataset</li>
<li>Preprocess (you will do this in Q3.2)</li>
<li>Split the data into x_train, y_train, x_test, y_test</li>
<li>Train the classifier on x_train and y_train</li>
<li>Predict on x_test</li>
<li>Evaluate testing accuracy by comparing the predictions from step 5 with y_test.</li>
</ol>
Here is an <a href="https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html">example</a><a href="https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html">.</a> Scikit has many other examples as well that you can learn from.

&nbsp;
