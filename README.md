# Ghanaian-Stock-Market-Prediction-using-the-Hidden-Markov-Model
An approach to try and predict the Ghanaian Stock Market using the Hidden Markov Model.
This approach seek to calculate the fractional change within day to day sock market prices.
After which the log likelihood of the current day is calculated. 
Subsequent days log likelihood are also calculated and the once close to the current day is slected and the difference is computed.
The data is then returned to the dataset and the window is shited to the next day for it log likelihood to be calculated.
After which the difference in fractional change is then calculated as a whole to get a slected days prediction.
