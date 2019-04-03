#https://www.kaggle.com/sebinvinicent/multi-text-classification-using-resumes-dat-943d53/edit


from pyexpat import features

import pandas as pd
df = pd.read_csv('resume_dataset.csv')
df.head()




#data cleaning module
#from io import StringIO


col = ['Category', 'Resume']
df = df[col]
df = df[pd.notnull(df['Resume'])]
df.columns = ['Category', 'Resume']
df['category_id'] = df['Category'].factorize()[0]
category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Category']].values)




#imbalanced classes visualisation

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('Category').Resume.count().plot.bar(ylim=0)
plt.show()


#Text repressentation as vectors

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Resume).toarray()
labels = df.category_id
features.shape

from sklearn.feature_selection import chi2
import numpy as np
N = 2
for Category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  # print("# '{}':".format(Category))
  # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  # print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# Multi-Class Classifier: Features and Design


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['Resume'], df['Category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# following 'software_engineer is variable of string which is resume for testing

software_engineer = '''﻿Irshad Ali
Email Address: irshadali18@gmail.com,irshadali@outlook.com 
Cell: 0321-7588568

Objective
I am looking forward to join a progressive organization. I am Strong team builder and leader. I have high level of personal morals and integrity. I am Goal oriented, self-motivated and committed to the successful outcome of the project. I am willing to work hard and have a great desire to learn.
Summary
    • Since March 2007, have 6 years plus of extensive hands on experience of website development.
    • An experienced team lead and team player with excellent communication and interpersonal skills who has the ability to work independently under pressure.
    • Currently working as Senior Software Engineer/Team Lead at Hashe Computer Solutions.
    • Masters in 2007 from the University College of Information Technology, Lahore, Pakistan.
Skills
Languages/Web Development
PHP, C# .Net, JavaScript, HTML, CSS, Java, XML, SQL
Frameworks
AJAX, Zend, Symfony2,  CodeIgniter
Open Source
Wordpress, Joomla, XCart, CSCart
Databases
MySQL, Oracle
Tools
Netbeans, Dreamweaver, SqlYog, NavicoSoft, MicroOlap


Experience
Hashe Computer Solution, Lahore, Pakistan 
Senior Software Engineer / Team Lead
(March 2008 – To date)
Responsibilities include team management, direct client communication and software development.
Mechtechnologies, Lahore, Pakistan 
Software Engineer
(March 2007 – February 2008)

Projects
Freight Ordering System – Hashe Computer Solutions
Role:
    • Development Lead
Tools:
    • PHP, MySQL, Ajax, JQuery , Web Services
Details:
    • This is a web based system, which provides an online competitive freight quotes within zip code range from best courier & transport companies of the region with favorable discounts and transit days. Later shipment can be booked out of these quotes and tracked though website. Companies can add / manage their locations, product catalog for swift use of the system.
    • This application works with SMC3 to acquire shipment rates for given locations and then apply different accessorial and fuel charges added by admin to calculate final shipment rates.
    • Using the back office application, admin can set different accessorial, discounts, fuel charges, and FAK classes for each company and carrier. Manage the Sales Representatives & this commission for different companies & Carriers.
    • Comprehensive report system provides reports about the shipment, carrier, customer, sale representative commission and billing reports. 
    • Complete Accounting System.

http://www.freightanywhere.com
http://www.tech-logistics.com

Online Golf Course Booking System – Hashe Computer Solutions
Role:
    • Application Developer
Tools:
    • PHP, MySQL, Ajax, JQuery, Web Services
Details:
    • This is a web based system, which provides golfers an easy way to use website to search, compare, and book golf tee times for free. With numerous golf courses available for play in most areas.

http://www.back9booking.com

Ecommerce Shopping System – Hashe Computer Solutions
Role:
    • Application Developer/Team Lead
Tools:
    • PHP, MySQL, JQuery, Zend
Details:
Complete ecommerce shopping system with following features
    • Administration system to easily update all product details, prices, pictures, stock details and other information online.
    • Manage Customer Accounts
    • Manage Wish list
    • Customer Reviews & Rating
    • Manage categories and products
    • Manage Product options and related products
    • Advanced pricing algorithms
    • Order and Invoice history
    • Take payments online using PayPal
    • Shopping cart system to allow easy purchase of products
    • Automatic email notification of orders
    • Full checkout procedure
    • Fast and friendly quick search and advanced search features
    • Reports of site visits, pages viewed, most viewed products, most ordered products and most viewed categories
http://www.tcig.co.uk

Free Home Listing – Hashe Computer Solutions
Role:
    • Application Developer
Tools:
    • PHP, MySQL, JQuery, Codeigniter
Details:
This is a property portal with three access level
    • Customer Login
        ◦ Search Properties by State And City, Key words and Zip Code with option in different miles radius i.e. search all properties having zip code 03055 and within 10 miles radius around it.  
        ◦ Register as Customer
        ◦ Manage their Listings
        ◦ Add/Edit property
        ◦ Add/Edit/Delete Properties Images
        ◦ Delete Properties
    • Agent Login
        ◦ Add/Edit property
        ◦ Add/Edit/Delete Properties Images
        ◦ Delete Properties
    • Admin login
        ◦ Manage Customers (Add/Edit/Delete/Active/Inactive)
        ◦ Manage Customer Packages
        ◦ Manage Agents (Add/Edit/Delete/Active/Inactive)
        ◦ Manage Listings (Add/Edit/Delete/Active/Inactive)
http://demo.hashe.com/freehomelistings/

Rockingham Acres – Hashe Computer Solutions
Role:
    • Application Developer
Tools:
    • PHP, MySQL, JQuery
Details:
This is an Online Flower Store has
    • Online Shopping Cart
    • Word Press Blog
http://www.rockinghamacres.com/


Third Coast Collection – Hashe Computer Solutions
Role:
    • Application Developer
Tools:
    • PHP, MySQL, JQuery
Details:
This website has
    • Online Shopping Cart
    • Authorized .Net Payment Integration
    • Word Press Blog
http://www.thirdcoastcollection.com/

PPA-Office Management System – Hashe Computer Solutions
Role:
    • Application Developer
Tools:
    • PHP, MySQL, JQuery
Details:
PPA (Pakistan Progressive Associate) is  licensed  by  Ministry   of  Labor,  Manpower  and  Overseas   Employment ,   Government of  Pakistan for recruitment  of  manpower.  So PPA-Office Management System is developed to manage & integrate all PPA internal processes (i.e. client, contracts, jobs, job seeker registration, resume bank, recruitment process, and visa & departure process). We split this big system into following modules.
    • Office Workflow Management System Administration: This application will allow the administration to
        ◦ Manage Companies, Contracts
        ◦ Application Configurations
        ◦ Manage invoices
        ◦ Manage administrative expenses
            ▪ Advertisement costs
            ▪ Courier charges
            ▪ Misc. charges to be posted
    • Office Workflow Management System: This application will automate the recruitment process of PPA administration and will implement all the business processes hence allowing straight through processing of jobs. This application will have three separate work flows
        ◦ Pre Processing – Jobs management, Resume management and data entry, short listing, interview scheduling and execution, selection of candidates and forwarding for post-processing. 
        ◦ Post Processing
        ◦ Archiving
    • Online Client / Candidate Portal: This portal will allow
        ◦ PPA administration to manage advertisement jobs
        ◦ PPA affiliated companies to:
            ▪ Login into the system
            ▪ Add jobs
            ▪ View list of candidates forwarded by PPA administration, short list them, add notes
            ▪ Browse/Search (if allowed) resume database, create resume lists, add notes on resumes
        ◦ Potential candidates to:
            ▪ Register
            ▪ Add resumes
            ▪ Search for jobs
    • System will allow the printing of all documents required during the execution of a case. System will allow three types of print
        ◦ Printing with PPA logo
        ◦ Printing without PPA logo – to be printed on PPA letter head
        ◦ Custom printing

NetSignNews.com – Hashe Computer Solutions
Role:
    • Development Lead
Tools:
    • PHP, MySQL
Details:
    • Net Sign News is a specialized news channel for with hearing disabilities. NetSignNews.com is an online news portal for NetSignNews. News videos are streamed on demand using FLV format files. This application has a power administration utility using which administrator can manage the contents being published on the website.

VegaPrint.co.uk – VegaSoft Technologies
Role:
    • Development Lead (Freelance)
Tools:
    • PHP, MySQL
Details:
    • This is print media service provider’s website. Here user can order print media products by paying online payment through PayPal, users can also track there orders online. 
    • Using the back office application, admin can add different products, services, special offers, shipment charges, manage users and orders. 

Bug Tracking – Mechtechnologies
Role:
    • Development Team Member
Tools:
    • PHP, MySQL
Details:
This is a web based application which allows software developers to track new bugs, prioritize and assign bugs to team members, generate bug reports, send email messages between users, attach files, customize the account according to their special needs and more.

Academic Projects
Student Information System - MIT Final Project
    • Student Information System superior University Lahore is a web based application developed in PHP and MySQL as database.


Education
Punjab University College of Information Technology, Lahore, Pakistan 
MSC Information Technology 
Year: 2007

Certifications
Microsoft Technologies (Exam: 70-480)
Microsoft Certified Professional 
Year: 2013
Microsoft Technologies (C# .Net)
EVS Lahore 
Year: 2013

Interests
Computer Gaming
References
References can be provided on request.'''



print(clf.predict(count_vect.transform([software_engineer])))

#model selection

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import cross_val_score
# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
# import seaborn as sns
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
#
#
# cv_df.groupby('model_name').accuracy.mean()



#model evaluation

# model = LinearSVC()
# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# from sklearn.metrics import confusion_matrix
# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
#
#
# from sklearn import metrics
# print(metrics.classification_report(y_test, y_pred, target_names=df['Category'].unique()))















