Hi there! Welcome to our team's project - Predicting Airline Prices!

In the modern era, where traveling has become a fundamental aspect of our lives, finding the most affordable flight tickets is crucial. The cost of flights fluctuates based on various factors, such as the airlines, timing of booking, departure schedules, and seating classes. For those looking to navigate their travels economically and efficiently, grasping the complex pricing patterns of airline tickets is essential. This project sets out to explore the complex dynamics behind flight pricing, utilizing an extensive flight dataset. Our goal is to not only predict the prices of flights with precision but also to discover insights that will aid travelers in their search for cost-effective and convenient travel solutions. This project utilizes a dataset which encompasses 300,261 unique flight booking alternatives, covering a timeframe of 50 days, from February 11th to March 31st, 2022. It has been compiled to facilitate data analysis and the development of predictive models for accurate flight price forecasting.

Hence, we came up with the Problem Statement.

Problem Statement: How do factors such as the choice of airline, timing of ticket purchase, departure and arrival times, variation, source and destination, and the class of service influence the pricing of airline tickets?

To structure our investigation effectively, we've segmented our primary question into **six** distinct sub-questions, each designed to explore a specific aspect of our overall inquiry:

a) Are ticket prices influenced by the flight's departure and arrival times? 

b) Is there a variation in ticket prices across different airlines? 

c) In what ways do ticket prices differ between Economy and Business class? 

d) How does altering the flight's origin and destination affect the price? 

e) Does a flight's duration play a role in varying the price?

f) Does the number of remaining days play a role in varying the price of flight tickets?

1) Data Cleaning and Transformation
   
Before proceeding with our exploratory analysis, we first drop useless columns that are not relevant or useful to researching our problem statement.

Columns to drop:
1) 'Unnamed: 0'
2) 'ch_code'
3) 'dep_time'
4) 'num_code'

We are also using Label encoding to transform categorical variables into numerical values.

Features
The various features of the cleaned dataset are explained below:

Airline: This categorical feature denotes the name of the airline company, with six distinct airlines.

Flight: Another categorical feature, it stores the flight code information.

Source City: Categorical, representing the city from which the flight originates, with six unique cities.

Departure Time: This derived categorical feature groups departure times into bins, offering six unique time labels.

Stops: Categorical, indicating the number of stops between the source and destination cities, with three distinct values.

Arrival Time: A derived categorical feature, grouping arrival times into bins with six distinct labels.

Destination City: City where the flight will land. It is a categorical feature having 6 unique cities.

Class: A categorical feature defining the seat class as either Business or Economy.

Duration: A continuous feature that displays the overall amount of time it takes to travel between cities in hours.

Days Left: This derived feature is calculated by subtracting the booking date from the trip date.

Price: The target variable, storing the ticket price information.

2) Exploratory Data Analysis
   
We explored the data by viewing which data is categorical and numerical based on its type. Next, we made a through numerical analysis by analysing its numerical variables, duration, days left and price.

We gathered that:

Average Price of flights dataset is 7425, and the maximum price are 123071 and the minimum price are 1105.

Average Days Left of flights is 26, and the maximum days left are 49 and the minimum days left are 1.

Average Duration of flights is 11.2, and the maximum duration is 49.8 and the minimum duration is 0.8.

We plotted a boxplot to analysise the median, mean and counted the outliers for each of the numerical variables. 
We gathered that:

Duration has 2110 outliers.

Days Left has 0 outliers.

Price has 123 outliers.

Next, we made a Skewness Analysis on the numerical variables, durations, days left and price. 

We gathered that: 

The  skewness values provide insights into the shape of the data distributions: 'duration' and 'price' exhibit right-skewed or positive skewed distributions, with longer tails on the right and data concentrated on the left. 'days_Left' has a skewness value close to zero, indicating a nearly symmetric distribution, which means the data is relatively balanced without strong skewness in either direction

Next, we made a Correlation Analysis between the three numerical variables, durations, day left and price. 

We gathered that:

There is little to no linear correlation between the three numerical variables.

We began our analysation on our problem statement, figuring out the relationship between ticket prices and other variables from the visualisation of airline ticket price data vs other variables.
a) Are ticket prices influenced by the flight's departure and arrival times?

Analysis: The box plot indicates that leaving late at night or arriving at night remains the most cost-effective option. It's also visible that arriving early morning is also cheap and afternoon flights are a bit cheaper than evening, morning and night flights.

b) Is there a variation in ticket prices across different airlines?

Analysis: Among the airlines, Air India and Vistara is having the most expensive flight tickets, whereas AirAsia provides the most affordable fares. Specifically, in the context of business class, Vistara's prices are the highest in comparison to AirAsia, SpiceJet, AirAsia, GO_First and Indigo seems to have around the same flight prices.

c) In what ways do ticket prices differ between Economy and Business class?

Analysis: Only two companies, Air India and Vistara, offer business flights, and there exists a substantial price difference between the two classes, with business tickets costing nearly five times as much as economy tickets. Business class typically offers enhanced services and amenities compared to economy class, such as more comfortable seating, premium meals, dedicated check-in counters, priority boarding, and additional space, which explains why there is a big gap in the prices of ticket between economy and business class.

d) How does altering the flight's origin and destination affect the price?

Analysis: It appears that flights departing from Delhi are frequently more affordable compared to those from other departure cities, likely due to the fact that as a capital city which is most likely to be larger and offers a greater variety of flights. Flights leaving and arriving at Bangalore seems to be highly priced. On the other hand, overall prices are relatively consistent, with Hyderabad emerging as the most expensive destination.

e) Does a flight's duration play a role in varying the price?

Analysis: It is visible in the above that with the increase in duration, ticket price also increases for both economy and business classes. This is likely due to the fact that long flights typically require more fuel consumption, which is a significant operational cost for airlines. As the flight duration increases, so does the fuel consumption, leading to higher operating expenses. Airlines often pass on these increased costs to passengers through higher ticket prices.

f) Does the number of remaining days play a role in varying the price of flight tickets?

Analysis: As we can see when compared to others when there are two days remaining for departure then the Ticket Price is very High for all airlines. The graph highlights how the prices rise slowly and then drastically start rising 20 days before the flight, but fall just one day before 
the flight up to three times cheaper. This pattern suggests that airlines may reduce ticket prices close to the departure date to fill empty seats and ensure high occupancy on their planes.

We begin the Machine Learning Portion of our project. Our primary objective is to build a predictive model that can accurately forecast the prices of flights. By predicting flight prices, you aim to provide valuable insights to travelers, airlines, travel agencies, and other stakeholders in the aviation industry Predicting flight prices can help travelers make informed decisions about booking flights, optimizing their travel budgets, and finding the best deals. In today's world, where travel has become an integral part of our lives, finding affordable flight tickets is essential. Factors like airline type, flight details, departure time, and more play a crucial role in determining ticket prices. Therefore, we'll leverage these variables to predict flight prices using machine learning models.

To kickstart our machine learning journey, we begin by preparing the predictor variables. We categorize them into two types: categorical and numerical data. Subsequently, we utilize label encoding for categorical features such as airline, flight, source city, departure and arrival times, stops, destination city, and class. Label encoding converts categorical data into a numerical format, a prerequisite for effective machine learning algorithm, enabling models to interpret categorical data during training and make predictions based on these features. It allows the models to capture relationships between different categories and flight prices.

To facilitate this process, we split our flights data into training and test sets in an 8:2 ratio. We then prepare our training data for further processing. As part of data preprocessing, we segment predictor variables into categorical and numerical types. We employ one-hot encoding and standard scaling techniques. Standardization ensures that all features are on the same scale, with a mean of 0 and a standard deviation of 1. This improves the performance of algorithms like SVM and logistic regression. One-hot encoding transforms categorical features into binary vectors, representing each category with a binary variable (0 or 1). It prevents ordinal relationships among categorical variables and ensures that the model interprets them correctly.

To predict airline prices accurately, we employ a diverse set of machine learning models, including Linear Regression, Random Forest, and Support Vector Machine (SVM). While Linear Regression serves as a foundational model, we'll pay special attention to the two newer models: Random Forest and SVM.

Before proceeding to select the best model among Linear Regression, Random Forest, and Support Vector Machine (SVM), we need to determine which one yields the highest performance. To accomplish this, we employ a process called hyperparameter tuning.
Hyperparameter tuning involves systematically searching for the optimal hyperparameters of a model to maximize its performance. We utilize GridSearchCV, a technique that exhaustively searches through a specified hyperparameter grid, evaluating each combination to identify the best-performing model. Once we've completed hyperparameter tuning, we obtain the best-optimized models for each algorithm. These optimized models are fine-tuned to achieve the highest predictive accuracy. With our optimized models in hand, we proceed to evaluate their performance. This evaluation involves assessing key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). By comparing these metrics across the three models, we can determine which one offers the most accurate predictions.

First, i will explain about the Support Vector Machine Learning Model. The Support Vector Machine (SVM) learning model was selected for predicting flight prices due to its effectiveness in handling high-dimensional data and its ability to capture intricate relationships between features. SVM is particularly well-suited for regression tasks like predicting airline prices because it excels in finding complex nonlinear patterns in data and can handle datasets with many features. SVM's strengths lie in its ability to handle high-dimensional data and capture complex relationships between features, making it a suitable choice for regression tasks like predicting airline prices. SVM relies on several key parameters to fine-tune its performance. Among these, C, kernel, and gamma are particularly crucial. C represents the regularization parameter in SVM. It controls the trade-off between achieving a low training error and a low testing error.  The kernel parameter determines the type of kernel function used in SVM, which defines the decision boundary between classes.  Lastly, gamma defines the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. It influences the decision boundary's flexibility, with higher gamma values leading to more complex decision boundaries, potentially resulting in overfitting. After hyperparameter tuning to optimize our SVM models, we proceeded to evaluate their performance. Unfortunately, the results were disappointing. The graph on the right illustrates the performance of our SVM model, which performed poorly. The negative R-squared value of negative 0.01 indicates that the model struggles to explain the variance in airline prices. Additionally, both Mean Absolute Error (MAE) and Mean Squared Error (MSE) are substantially higher, signifying significant prediction errors. While SVM may be a powerful model in certain domains, such as image classification or text mining, it proved inadequate for our flights dataset. This is not desirable for machine learning, we want the machine learning tool to perform well enough to predict.

Hence to improve machine learning further, we carry out random forest.  We chose to employ this model to predict flight prices for several reasons. We chose Random Forest because of its proven track record in handling high-dimensional data, capturing complex relationships, and delivering accurate predictions. Random Forest is renowned for its robustness and ability to handle both categorical and numerical data effectively. In the context of our problem statement, where we have diverse features such as airline, departure time, and destination city, Random Forest's versatility makes it an ideal candidate for predicting flight prices. Firstly, we hypertune and obtains the best model using parameters, n_estimators which refers to the number of decision trees that are aggregated to form the Random Forest.  Max_depth which controls the maximum depth of each decision tree in the ensemble. min_samples_split which determines the minimum number of samples required to split an internal node. min_samples_leaf which specifies the minimum number of samples required to be at a leaf node. After hyperparameter tuning and obtaining the best models for Random Forest, we moved on to evaluate its performance. As depicted in the graph on the right, the Random Forest model excelled in performance. The R-squared value of 0.96 indicates a remarkable ability to explain the variance in airline prices. Moreover, both Mean Absolute Error (MAE) and Mean Squared Error (MSE) are substantially lower compared to the other models, signifying minimal prediction errors.
In summary, Random Forest emerges as a powerful tool for predicting airline prices, offering high accuracy, robustness, and the ability to handle diverse features effectively. Its performance in explaining the variance in flight prices, as evidenced by the high R-squared value and low prediction errors, underscores its suitability for our problem statement.

In conclusion, based on these results:

The Random Forest model appears to be the most suitable for predicting airline prices, as it achieves the highest R-squared value and lowest prediction errors (MAE and MSE). Linear Regression also performs reasonably well but exhibits higher prediction errors compared to Random Forest. The Support Vector Machine model performs poorly in predicting airline prices compared to the other models. Therefore, as our goal is to accurately predict airline prices, the Random Forest model would be the preferred choice among the models evaluated in this analysis.

As such, we begin training the Random forest model using the best hyperparameters we obtained earlier. After training, we make predictions on the testing data using the trained model to obtain insights into regression. We obtained an R2 value of 0.98, an MAE of 1329, and an MSE of 9642796.

In conclusion, the Random Forest model demonstrates exceptional performance in predicting airline prices, offering high accuracy and minimal prediction errors. Its robustness, flexibility, and ability to handle diverse features make it an ideal choice for our problem statement. By leveraging Random Forest, we can provide valuable insights into pricing dynamics in the airline industry and facilitate informed decision-making for both travelers and airlines.

From this project, we've gained invaluable experience in utilizing various machine learning techniques, including Random Forest and Support Vector Machine, for regression to accurately predict prices. Furthermore, we've mastered the art of employing cross-validation grid search to meticulously optimize the selection of the best model for price prediction.

With our newfound knowledge, we've developed a robust framework that enables us to forecast airline prices based on a multitude of influencing factors. This capability empowers us to anticipate fluctuations in airline ticket prices and make well-informed decisions when planning trips to different destinations.

Equipped with our sophisticated ML model, we can navigate the ever-changing landscape of airline pricing with confidence. By leveraging these insights, we can make strategic choices that enhance both our travel experiences and financial efficiency. Whether it's booking flights well in advance or identifying the most cost-effective travel times, our predictive model provides us with the tools to travel smarter and more effectively.

To conclude, here are our data-driven insights.

1) The RandomForest performs best on the test dataset, achieving an R^2 score of 0.9812 and an MAE score of 1329.
2) Vistara and AirIndia emerge as the priciest companies, while AirAsia offers the most budget-friendly options. However, for business tickets, only Vistara and AirIndia are available, with Vistara being marginally pricier. Airlines may differentiate themselves based on the level of service and amenities provided onboard, such as seating comfort, in-flight entertainment, meals, and customer service. Higher-priced airlines often offer more luxurious or comprehensive services compared to budget airlines like AirAsia, which may reflect in their ticket prices.
3) Generally, ticket prices exhibit gradual increases until 20 days before the flight, after which they escalate dramatically. However, the day before the flight often sees unsold seats, allowing for tickets to be found at one-third of the price compared to the previous day. This pattern suggests that airlines may reduce ticket prices close to the departure date to fill empty seats and ensure high occupancy on their planes.
4) Ticket prices tend to rise with the duration of the flight. This is likely due to the fact that long flights typically require more fuel consumption, which is a significant operational cost for airlines. As the flight duration increases, so does the fuel consumption, leading to higher operating expenses. Airlines often pass on these increased costs to passengers through higher ticket prices.
5) Regarding the flight time: Departures during the afternoon and late night tend to be cheaper, while nighttime departures are pricier. This might be because, many travelers prefer not to depart during the late-night or early-morning hours due to inconvenience or discomfort, leading to decreased demand and lower prices during these times.
6) Regarding the cities of the trip: Flights departing from Delhi are the most economical, while those from other cities exhibit comparable prices, slightly favoring Chennai. Flights to Delhi are the most affordable, whereas those to Bengaluru are the costliest. Generally, flights with more stops tend to have higher ticket prices.









