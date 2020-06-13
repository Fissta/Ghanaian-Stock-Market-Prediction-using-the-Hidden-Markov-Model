"""
Usage: analyse_data.py --company=<company>
"""
import sys
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from docopt import docopt

# args = docopt(doc=__doc__, argv=None, help=True,
#   version=None, options_first=False)
# company = pd.read_csv('AAPL.csv')
# Supress warning in hmmlearn
warnings.filterwarnings("ignore")


# Change plot style to ggplot (for better and more aesthetic visualisation)
# plt.style.use('ggplot')


class StockPredictor(object):
    def __init__(self, company=[], n_hidden_states=range(2, 5), test_size=0.3,
                 n_latency_days=20, likelihood_vect=np.empty([0, 1]),
                 aic_vect=np.empty([0, 1]), bic_vect=np.empty([0, 1]),
                 n_iter_number=1000,
                 n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10):

        self.company = company
        self.n_latency_days = n_latency_days

        self.model = hmm.GaussianHMM(n_components=n_hidden_states)

        self.n_hidden_states = n_hidden_states

        self._split_train_test_data(test_size)

        self.likelihood_vect = likelihood_vect

        self.aic_vect = aic_vect

        self.bic_vect = bic_vect

        self.n_iter_number = n_iter_number

        self._compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def _split_train_test_data(self, test_size):
        for company in self.company:
            data = pd.read_csv('{company}.csv'.format(company=company))
            _train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

            self._train_data = _train_data
            self._test_data = test_data

    # Calculating Mean Absolute Percentage Error of predictions
    def calc_mape(predicted_data, true_data):
        return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])

    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['Open'])
        close_price = np.array(data['Close'])
        high_price = np.array(data['High'])
        low_price = np.array(data['Low'])

        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))
        #return np.column_stack((open_price, close_price, high_price, low_price))

    def number_hidden_states(self):
        likelihood_vect = np.empty([0, 1])
        aic_vect = np.empty([0, 1])
        bic_vect = np.empty([0, 1])
        dataset = StockPredictor._extract_features(self._train_data)

        for states in self.n_hidden_states:
            num_params = states ** 2 + states
            dirichlet_params_states = np.random.randint(1, 50, states)
            # model = hmm.GaussianHMM(n_components=states, covariance_type='full', startprob_prior=dirichlet_
            # params_states, transmat_prior=dirichlet_params_states, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
            model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=self.n_iter_number)
            model.fit(dataset)
            if model.monitor_.iter == self.n_iter_number:
                print('Increase number of iterations')
                sys.exit(1)
            likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
            aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
            bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) + num_params
                                  * np.log(dataset.shape[0])))

        opt_states = np.argmin(bic_vect) + 2
        # print('Optimum number of states are {}'.format(opt_states))

        return opt_states

    def fit(self):
        print('Optimum number of states are {}'.format(self.number_hidden_states()))
        for company in self.company:
            dataset = StockPredictor._extract_features(self._train_data)

            for idx in reversed(range(self.n_iter_number)):
                train_dataset = dataset
                # train_dataset = train_dataset.reshape(1) model = hmm.GaussianHMM(n_components=opt_states,
                # covariance_type='full', startprob_prior=dirichlet_params, transmat_prior=dirichlet_params,
                # tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
                if idx == self.n_iter_number - 1:
                    self.model = hmm.GaussianHMM(n_components=self.number_hidden_states(), covariance_type='full',
                                            tol=0.0001,
                                            n_iter=self.n_iter_number,
                                            init_params='stmc')
                else:
                    # Retune the model by using the HMM paramters from the previous iterations as the prior
                    self.model = hmm.GaussianHMM(n_components=self.number_hidden_states(), covariance_type='full',
                                            tol=0.0001,
                                            n_iter=self.n_iter_number,
                                            init_params='')
                    self.model.transmat_ = transmat_retune_prior
                    self.model.startprob_ = startprob_retune_prior
                    self.model.means_ = means_retune_prior
                    self.model.covars_ = covars_retune_prior

                self.model.fit(train_dataset)

                transmat_retune_prior = self.model.transmat_
                startprob_retune_prior = self.model.startprob_
                means_retune_prior = self.model.means_
                covars_retune_prior = self.model.covars_

    def _compute_all_possible_outcomes(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)

        self._possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))

    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_start_index: previous_data_end_index]
        previous_data_features = StockPredictor._extract_features(previous_data)

        outcome_score = []

        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((possible_outcome, previous_data_features))
            outcome_score.append(self.model.score(total_data))

        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        open_price = self._test_data.iloc[day_index]['Open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predicted_frac_change)

    # Calculating Mean Absolute Percentage Error of predictions
    def calc_mape(predicted_data, true_data):
        return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0]) * 100

    def predict_close_prices_for_days(self, days, with_plot=False):
        predicted_close_prices = []

        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))


        test_data = self._test_data[0: days]
        days = np.array(test_data['Date'])
        actual_close_prices = test_data['Close']

        if with_plot:

            plt.figure(figsize=(16, 14))
            plt.plot(days, predicted_close_prices, 'k-', label='Predicted price');
            plt.plot(days, actual_close_prices, 'r--', label='Actual  price')
            plt.xlabel('Time steps')
            plt.ylabel('Price')
            plt.title('{company}'.format(company=self.company))
            plt.legend(loc='upper left')

            plt.show()

        mape = StockPredictor.calc_mape(predicted_close_prices, actual_close_prices)
        print(mape)

        return predicted_close_prices, mape


stock_predictor = StockPredictor(company=['total'])
stock_predictor.fit()
#stock_predictor._get_most_probable_outcome(200)
stock_predictor.predict_close_prices_for_days(300, with_plot=True)
