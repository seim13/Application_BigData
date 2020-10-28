<h1> Application_BigData <h1>

<h2> Part 1 of the project for Advanced BigData Systems 2020-2021 <h2>
<br>
Realised by Gilles Robin and Juliette Seimandi
 
<br>

<h3> Goals of the Project <h3>
<br>
 <p> The goal of this project is to apply some concepts & tools seen in the 3 parts of this course, this project is organized into 3 parts :</p>
<ol>
<li> Part 1 : Building Classical ML projects with respect to basic ML Coding best practices </li>
  
<li> Part 2 : Integrate MLFlow to your project </li>
  
<li> Part 3 : Integrate XAI (ML Interpretability with SHAP) to your project </li>

</ol>

<h3> Part 1 </h3>

<p>We use a DataSet of Home credit risk classification in this project, you can find this dataset at https://www.kaggle.com/c/home-credit-default-risk/data </p>
  
<p>Download application_train.csv and application_test.csv and put it in your home directory with allother file</p>

<p>You can find our load an analyse of the data in the <strong><em> Import data and data cleaning.ipynb </strong></em> then our feature engineering in the <strong><em> Feature engineering.ipynb </strong></em> with the creation of new and clean data set that will be used in our machine learning model</p>

<p>Then we build Xgboost, Random Forest and Gradient Boosting and improve all model in the file <strong><em> Model building For balanced dataset.ipynb </strong></em> </p> 
 <p> You can see our parameter in the <strong><em>applicationOfBDEnvironment.yml file</em></strong> </p>

<br>
<h3> Sphinx Documentation </h3>
<br>
 <p>To get a better view of our model you can check our Sphinx documentation, to do so :</p>
<br>
<ul>
<li>  Clone this git repository </ li>
  <br>
<li>  launch a terminal </li>
  <br>
<li> go to the git repository and launch the command <em> open build/html/index.html </em> </li>
 </ul>
   <br>
 <p>Then you will be able to understand all our project.</p>
  
<h3> Part 2 </h3>
<p> In this part we introduce MLFlow to our project, in order to be able to track the parameters of our models.</p>
<p> You an unzip it and see the ML models by using the command MLflow ui in a terminal. if you want to rexecute the python scripts you have to put them in the same folder as the one containing the csv created by running "Import data and data cleaning.ipynb" and Feature engineering.ipynb.   </p>
<h3> Part 3 </h3>

<p> We have to use SHAP library to have a better understanding of our performance. You can find our shap implemantation in the file <strong><em>XAI with SHAP method.ipynb</strong></em>  </p> 
<p> Infortunatly we cannot visualise our graph in the sphinx documentation because it doesn't compile the java script package but if you run the file on your own computer you will be able to have them</p>


