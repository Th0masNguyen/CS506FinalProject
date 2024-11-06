# CS506FinalProject
Video link:  
https://youtu.be/VVUz1f2MdP4

Description:   
I would like to create a project that analyzes historic Star Market Weekly Ad data in order to help me make optimal grocery shopping decisions.

Goal(s):   
Successfully predict the future occurrences of various items in the Star Market Weekly Ad. In particular, I only really care about predicting when ground beef or boneless chicken breasts will next go on sale. So as a time-to-event prediction problem, I want to be able to feed the model (item name) so that it will return to me (next sale date).

Data Collection:  
I have manually extracted data from various online archives of images of the Star Market weekly ads. It’s a bit over a year-and-a-half’s worth of consecutive weekly ads. The rows of the data take the form of (Week, Name, Category, Regular Price, Sale Price). For instance, (10/18/2024, ground beef, meat, 4.99, 3.47). I decided to focus primarily on meat items since I felt that this was the only category that had sufficient availability and consistency on the weekly ads. 

Data Cleaning:  
Missing regular prices were filled in with the last recorded regular price. Some items such as “ground beef” were separated into “ground beef” and “fancy ground beef” because their original recording did not reflect the difference in regular prices. For now, I’m dropping everything besides Week and Name because the pricing trends don’t seem interesting or useful.

Feature Extraction:  
For each unique item name, I have WeeksSinceLastSale and WeeksUntilNextSale.

Data Modeling:  
Currently, I’m using a very basic linear regression model for each unique item name where the independent variable is WeeksSinceLastSale and the dependent variable is WeeksUntilNextSale.

Data Visualization:  
Not too sure yet. I’ll probably want a simple scatter plot where each point represents the presence of a discount for a certain food item across all of the weeks from the collected data. I also want to visualize the probability distributions of a given food item being on sale in the next week.

Test Plan:  
Testing on 4-5 instances of the weekly ads toward the end of October through November sounds good to me. Also, the lack of raw data volume makes testing a little tricky.

Next Steps:  
I kinda wanna see what the predictions would look like if I treated it as a classification problem instead of regression. I could probably try both. I also need to work on a web app interface.
