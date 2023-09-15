## Gas Turbine Monitoring System
app.py: This script is used to start Dash App on local host.

prediction.py: This script is used to predict NOx value from streaming sensor data.

Procfile: A file required for Heroku deployment.

runtime.txt: A file required for Heroku deployment.

requirements.txt: File that contains required Python packages for the project. (For Dash App, ML part may need additional libraries.)

static: This folder contains several static files that might be used by Dash App.

ML: This folder contains all files that are related to data analysis workflow. EDA, Feature Engineering, Model Development parts of the project are carried out in main.py

While working with Dash App project root folder should be the whole folder. However, root folder should be changed to ML while working with scripts in ML folder.