#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset - [Soccer Database]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This soccer database comes from Kaggle and is well suited for data analysis and machine learning. It contains data for soccer matches, players, and teams from several European countries from 2008 to 2016. This dataset is quite extensive, and we encourage you to read more about it here.
# 
# The database is stored in a SQLite database. You can access database files using software like DB Browser.
# This dataset will help you get good practice with your SQL joins. Make sure to look at how the different tables relate to each other.
# Some column titles should be self-explanatory, and others you’ll have to look up on Kaggle.
# 
# 
# ### Question(s) for Analysis
# >**Tip**: Clearly state one or more questions that you plan on exploring over the course of the report. You will address these questions in the **data analysis** and **conclusion** sections. Try to build your report around the analysis of at least one dependent variable and three independent variables. If you're not sure what questions to ask, then make sure you familiarize yourself with the dataset, its variables and the dataset context for ideas of what to explore.
# 
# > **Tip**: Once you start coding, use NumPy arrays, Pandas Series, and DataFrames where appropriate rather than Python lists and dictionaries. Also, **use good coding practices**, such as, define and use functions to avoid repetitive code. Use appropriate comments within the code cells, explanation in the mark-down cells, and meaningful variable names. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
hz=pd.read_csv("noshowappointments-kagglev2-may-2016.csv")
hz.head()


# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[4]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
hz.shape


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 
# 
# 
# 
# > **Tip**: - Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.
# 
# 
# ### Research Question 1 (Replace this header name!)

# In[8]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
hz.duplicated().sum()


# ### Research Question 2  (Replace this header name!)

# In[9]:


# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.
hz['PatientId'].nunique()


# In[10]:


hz['PatientId'].duplicated().sum()


# In[11]:


hz.duplicated(['PatientId','No-show']).sum()


# In[13]:


hz.info()


# In[14]:


hz.describe()


# In[42]:


hz.query("Age==-1")


# In[43]:


hz.drop(index=99832,inplace=True)


# In[44]:


hz.describe()


# In[51]:


hz.rename(columns={'Hipertension':'Hypertension'},inplace=True)
hz.rename(columns={'No-show':'No_show'},inplace=True)
hz.head()


# In[54]:


hz.drop_duplicates(['PatientId','N0_show'],inplace=True)
hz.shape


# In[55]:


hz.drop(['PatientId','AppointmentID','ScheduledDay','AppointmentDay'],axis=1,inplace=True)
hz.head()


# In[58]:


hz.hist(figsize=(16.5,6.5));


# In[59]:


show=hz.N0_show=="No"
noshow=hz.N0_show=="Yes"
hz[show].count(),hz[noshow].count


# In[61]:


hz[show].mean
hz[noshow].mean


# In[63]:


def attendance(hz,col_name,attendent,absent):
    plt.figure(figsize=[16,4])
    hz[col_name][show].hist(alpha=.5,bins=10,color='blue',label='show')
    hz[col_name][noshow].hist(alpha=.5,bins=10,color='red',label='noshow')
    plt.legend();
    plt.title("compare according to age")
    plt.xlabel('Age')
    plt.ylabel('Patients numbers');
attendance(hz,'Age',show,noshow)    
    
    


# In[68]:


plt.figure(figsize=[16,4])
hz[show].groupby(['Hypertension','Diabetes']).mean()['Age'].plot(kind='bar',color='blue',label='show')
hz[noshow].groupby(['Hypertension','Diabetes']).mean()['Age'].plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title("compare according to chronic diseases")
plt.xlabel('chronic diseases')
plt.ylabel('mean Age'); 


# In[70]:


hz[show].groupby(['Hypertension','Diabetes']).mean()['Age'],hz[noshow].groupby(['Hypertension','Diabetes']).mean()['Age']


# In[71]:


def attendance(hz,col_name,attendent,absent):
    plt.figure(figsize=[16,4])
    hz[col_name][show].value_counts(normalize=True).plot(kind='pie',label='show')
    plt.legend();
    plt.title("compare attendance and Gender")
    plt.xlabel('Gender')
    plt.ylabel('Patients numbers');
attendance(hz,'Gender',show,noshow)    


# In[75]:


plt.figure(figsize=[16,4])
hz[show].groupby('Gender').mean()['Age'].plot(kind='bar',color='blue',label='show')
hz[noshow].groupby('Gender').mean()['Age'].plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title("compare according to chronic diseases")
plt.xlabel('chronic diseases')
plt.ylabel('mean Age'); 


# In[79]:


hz[show].groupby('Gender').mean()['Age'],hz[noshow].groupby('Gender').mean()['Age']


# In[80]:


hz[show].groupby('Gender').Age.median(),hz[noshow].groupby('Gender').Age.median()


# In[84]:


def attendance(hz,col_name,attendent,absent):
    plt.figure(figsize=[16,4])
    hz[col_name][show].value_counts(normalize=True).plot(kind='pie',label='show')
    plt.legend();
    plt.title("compare according to SMS_recieved")
    plt.xlabel('SMS')
    plt.ylabel('Patients numbers');
attendance(hz,'SMS_received',show,noshow)    


# In[88]:


plt.figure(figsize=[16,4])
hz.Neighbourhood[show].value_counts().plot(kind='bar',color='blue',label='show')
hz.Neighbourhood[noshow].value_counts().plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title("compare according to Neighbourhood")
plt.xlabel('Neighbourhood')
plt.ylabel('Patients numbers');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Tip**: Finally, summarize your findings and the results that have been performed in relation to the question(s) provided at the beginning of the analysis. Summarize the results accurately, and point out where additional research can be done or where additional information could be useful.
# 
# > **Tip**: Make sure that you are clear with regards to the limitations of your exploration. You should have at least 1 limitation explained clearly. 
# 
# > **Tip**: If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[89]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




