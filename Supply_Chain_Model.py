# Update from previous code:
# ADD customer satisfaction cost to the main cost
# Simulation model is completed
# SA for Neural Network
import numpy as np
from itertools import product
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv1D, Dropout , Flatten, MaxPool1D, LeakyReLU
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
import keras.backend
from smt.sampling_methods import LHS
from scipy.optimize import dual_annealing
import pprint


warnings.filterwarnings("ignore")

class main_model:
    def __init__(self, num_run, num_period, num_products, invent_coeffs, prod_demand_var):
        self.num_runs = num_run
        self.num_prd = num_period
        self.num_products = num_products
        self.inventory_coeff = invent_coeffs
        self.prod_dmnd_var = prod_demand_var
        self.model_parameters()


    def model_parameters(self):
        self.num_facility = 3
        self.num_regions = 8
        self.production_cost = np.array([0.3, 0.5, 0.4, 0.2, 0.4])
        self.transp_cost = 0.02
        self.inventory_cost = np.array([0.4, 0.3, 0.6, 0.2, 0.3])
        self.outsource_cost = np.array([0.8, 0.7, 1.2, 0.9, 1.1])
        self.losecust_penalty = np.array([1.5, 2, 2, 1, 2])
        self.fac_reg_distance = np.array([[32,21,19,31,22,35,18,33],
                                 [25,36,30,24,16,14,31,19],
                                 [18,12,27,11,33,28,23,22]])
        self.max_order = 40
        self.min_order = 1
        self.demand_product_region = np.array([[12,12,7,11,8,15,10,15],
                                      [10,13,11,5,8,13,11,8],
                                      [15,12,16,20,13,13,7,13],
                                      [6, 11,7,8,10,7,10,7],
                                      [18,14,18,13,18,13,16,18]])


        self.M = 10 ** 4
        self.satisfactory_thresh = [0.9]* self.num_products


    def demands(self):
        demands = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_regions))
        for i in range(self.num_runs):
            for j in range(self.num_prd):
                for k in range(self.num_products):
                    for c in range(self.num_regions):
                        demands[i,j,k,c] = np.round(np.random.normal(self.demand_product_region[k,c], np.sqrt(self.prod_dmnd_var[k])))
        demands[demands<0]=0

        return np.array(demands)


    def simulation(self, ):
        demands = self.demands()
        target_inventory = np.zeros((self.num_products, self.num_facility))
        produced_level = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_facility))
        transported_level = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_facility))
        outsourced_level = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_regions))
        y_count = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_regions))
        inventory_level = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_facility))

        produced_cost = []
        transported_cost = []
        outsourced_cost = []
        inventory_cost = []

        for i in range(self.num_facility):
            base = 0
            for j in range(3):
                low = 1000
                index = 0
                for k in range(self.num_regions):
                    if self.fac_reg_distance[i][k] < low and self.fac_reg_distance[i][k] > base:
                        low = self.fac_reg_distance[i][k]
                        index = k
                target_inventory[:, i] += self.demand_product_region[:, index]
                base = low


        for j in range(self.num_facility):
            target_inventory[:,j] = np.multiply(target_inventory[:,j], self.inventory_coeff)

        for i in range(self.num_runs):
            for t in range(self.num_prd):
                for k in range(self.num_products):
                    for j in range(self.num_facility):
                        if inventory_level[i,t,k,j] < target_inventory[k,j]:
                            produced_level[i,t,k,j] = target_inventory[k,j] - inventory_level[i,t,k,j]
                            inventory_level[i,t,k,j] += produced_level[i,t,k,j]

                for c in range(self.num_regions):
                    base = 0
                    for m in range(self.num_facility):
                        low = 1000
                        index = 1
                        for j in range(self.num_facility):
                            if self.fac_reg_distance[j,c] < low and self.fac_reg_distance[j,c]>base:
                                low = self.fac_reg_distance[j,c]
                                index = j

                        for k in range(self.num_products):
                           if demands[i,t,k,c] > inventory_level[i, t,k, index] and demands[i,t,k,c] > 0:

                               demands[i,t,k,c] -= inventory_level[i,t,k, index]

                               transported_level[i,t,k, index] = inventory_level[i,t,k, index]
                               inventory_level[ i, t, k, index] = 0
                           elif demands[i,t,k,c] <= inventory_level[i,t, k, index] and demands[i,t,k,c] > 0:
                               transported_level[i,t, k, index] = demands[i,t,k,c]
                               demands[i,t,k,c] = 0
                               inventory_level[i,t, k, index] -= transported_level[i,t, k, index]
                               if t != self.num_prd - 1:
                                   inventory_level[i,t+1, k, index] = inventory_level[i,t, k, index]
                        base = low

                    for k in range(self.num_products):
                        if demands[i,t,k,c] > 0:
                            outsourced_level[i,t, k, c] = demands[i,t,k,c]
                            if outsourced_level[i,t, k, c]>0:
                                y_count[i,t,k,c] += 1
                            demands[i,t,k,c] = 0

        pr_region = 0
        for i in range(self.num_runs):
            for j in range(self.num_prd):
                pr_region += y_count[i,j]

        pr = np.sum(pr_region,axis=1)

        customer_dissatis = pr/(self.num_runs*self.num_prd*self.num_regions)
        print("Customer Dissatisfaction for each Product\n",customer_dissatis)


        for i in range(self.num_runs):
            for t in range(self.num_prd):
                produced_cost.append(np.dot(produced_level[i,t,...].sum(axis=1), self.production_cost))
                transported_cost.append(np.dot(transported_level[i,t,...].sum(axis=1), np.repeat(self.transp_cost, self.num_products)))
                outsourced_cost.append(np.dot(outsourced_level[i,t,...].sum(axis=1), self.outsource_cost))
                inventory_cost.append(np.dot(inventory_level[i,t, ...].sum(axis=1), self.inventory_cost))




        mean_cost = (np.array(inventory_cost).sum() + np.array(outsourced_cost).sum() +
                np.array(produced_cost).sum() + np.array(transported_cost).sum())/self.num_runs

        print("Mean COst", mean_cost)
        # print("Customer dissatisfaction cost",self.M * customer_dissatis)

        cost_dis = [None]*len(self.satisfactory_thresh)
        print(cost_dis)
        # for i in range(len(self.satisfactory_thresh)):
        #     if customer_dissatis[i] >= (1 - self.satisfactory_thresh[i]):
        #         mean_cost += self.M * customer_dissatis[i]
        #         cost_dis[i] = self.M * customer_dissatis[i]
        #     else:
        #         mean_cost += 0
        #         cost_dis[i]= 0

        print("Dissat cost", cost_dis)
        print("Total Cost", mean_cost)

        # Mean_cost is the total cost
        return mean_cost


class ANN:
    def __init__(self, x_train, x_test, y_train, y_test, num_epochs):
        self.data(x_train, x_test)
        self.y_train = y_train
        self.y_test = y_test
        self.num_epochs = num_epochs
        # self.inputss = inputss


    def plot_progress(self, model):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)  # MSE #MAE #MAPE

        ax1.plot(model.epoch, model.history['root_mean_squared_error'], color='b')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Root Mean Squared Error")

        ax2.plot(model.epoch, model.history['mae'], color='r')
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("Mean Absolute Error")

        ax3.plot(model.epoch, model.history['loss'], color='c')
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("Mean Absolute Percentage Error")
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.55, hspace=0.1)

        plt.savefig("./fig_sim_3.png", dpi=300)

    def data(self, x_train, x_test):
        # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
        # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        self.x_train_data = x_train
        self.x_test_data = x_test

    def model(self, inputss):
        keras.backend.clear_session()
        model = Sequential()
        # model.add(InputLayer(input_shape=(self.x_train_data.shape[1])))
        model.add(Dense(10, input_dim=self.x_train_data.shape[1], activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        # model.summary()

        model.compile('adam', loss='mape', metrics=[RootMeanSquaredError(), 'mae'])
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=50, verbose=0)
        self.plot_progress(model_history)
        # print ("Input:",inputss)
        # print(inputss.shape)
        # print(len(inputss))
        xx = np.array([inputss])
        predicted = model.predict(xx)
        print (predicted)
        # print(self.inputss.shape)
        # print(x_train.shape)
        # accs = accuracy_score(predicted, y_test)
        # print('Accuracy is:', accs * 100)

        # _, accuracy = model.evaluate(x_test, y_test)
        # print('Accuracy: %.2f' % (accuracy * 100))

        return predicted[0][0]


def target_coeff(coeffs, number_products):
    perms = product(coeffs, repeat=number_products)
    out_coef = []
    for i in list(perms):
        out_coef.append(i)
    return np.array(out_coef)


def NN_data_generate(demand_variation):
    number_runs = 2  # 100
    number_periods = 3 # 52
    number_products = 5  # Arbitrary
    # demand_variation = [2, 2, 2, 2, 2]  # Arbitrary
    # cost_satis =
    # coeffs_train = 1
    coeffs_train = (0.5, 1, 1.5)
    number_coeffs_train = 3
    number_coeffs_test = 3
    coeffs_test = (0.5, 1, 1.5)
    target_inv_train = target_coeff(coeffs_train, number_products)
    # print(target_inv_train)
    # print(len(target_inv_train))
    target_inv_test = target_coeff(coeffs_test, number_products)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(number_coeffs_train**number_products):

        invenroty_coeffs_train = target_inv_train[i,:]
        x_train.append(invenroty_coeffs_train)
        model_instance_train = main_model(number_runs, number_periods, number_products, invenroty_coeffs_train, demand_variation)
        cost_train= model_instance_train.simulation()
        print("{}th iteration for train:{}".format(i+1,cost_train))
        y_train.append(cost_train)


    for i in range(number_coeffs_test ** number_products):
        invenroty_coeffs_test = target_inv_test[i, :]
        x_test.append(invenroty_coeffs_test)
        model_instance_test = main_model(number_runs, number_periods, number_products, invenroty_coeffs_test,
                                          demand_variation)
        cost_test = model_instance_test.simulation()
        print("{}th iteration for test:{}".format(i+1,cost_test))
        y_test.append(cost_test)


    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def LHSmp(xlimits, num): #LHS Sampling

    sampling = LHS(xlimits=xlimits)

    LHS_out = sampling(num)
    return LHS_out


if __name__ == "__main__":
    var_default = [100,100,100,100,100,100]


    cost = []
    satis = []
    runs_list = [5, 10, 20, 50, 100, 500, 1000]
    # runs_list = [5, 10, 20, 50, 100, 500, 1000, 10000]
    # for i_run in runs_list:
    #
    #     model_instance_train = main_model(i_run+1, 12, num_products=5, invent_coeffs=[1,1,1,1,1],
    #                                   prod_demand_var=var_default)
    #
    #     cost_train, customer_sati = model_instance_train.simulation()
    #     cost.append(cost_train)
    #     satis.append(customer_sati)

    satis = np.array(satis)
    # print(cost)
    # print(satis)
    # print(satis.shape)

    runs = np.arange(1,len(runs_list))
    # print(runs)
    # # print(cost)
    # plt.plot(runs_list, cost)
    # plt.show()

    satis = satis.T
    # print(satis)
    # Plot

    # print("stais 1:", satis[0])
    # for i in range(5):
    #     label = "product%d" %(i+1)
    #     plt.plot(range(len(runs_list)), satis[i], label = label)
    #
    # plt.xticks(range(len(runs_list)), runs_list)
    # # plt.xlim([5,10, 20, 50, 100, 500])
    # # plt.title("Customer Satisfaction Level Convergence")
    # plt.xlabel("Number of Simulation Replication")
    # plt.ylabel("Customer Satisfaction Rate")
    # plt.ylim(0.1, 0.35)
    # plt.legend()
    # plt.show()


    xlimits = np.array([[9, 16], [2, 10], [2, 10], [2, 10], [2, 10]])
    num = 100
    EnvLHS = LHSmp(xlimits, num)
    # print(EnvLHS)
    # print(type(EnvLHS))
    # print(len(EnvLHS))



    # a = ANN(x_train, x_test, y_train, y_test, num_epochs)
    # # inp1 = np.array([[1, 1, 1, 1, 1]])
    # inp1 = [1, 1, 1, 1, 1]

    # # model5 = a.model(x)
    #
    # # X, min_func = PSO_NN(model5).Pso_opt()
    # # X, min_func = PSO_NN(model5).Pso_opt()
    #



    # Ditriminstic without metamodel
    num_epochs = 100
    num_run = 100
    num_periods = 12
    num_products = 5
    inventory_coef = [1,1,1,1,1]
    var = [100, 100, 100, 100, 100]
    x_train, y_train, x_test, y_test = NN_data_generate(var)
    a = ANN(x_train, x_test, y_train, y_test, num_epochs)
    # inp1 = np.array([[1, 1, 1, 1, 1]])
    inp1 = [1, 1, 1, 1, 1]

    # model5 = a.model(x)

    # X, min_func = PSO_NN(model5).Pso_opt()
    # X, min_func = PSO_NN(model5).Pso_opt()

    func = lambda x: a.model(x)
    # print (x)
    # print(min_func)
    print("New value",func(inp1))
    lw = [0.1] * 5
    up = [3] * 5
    ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
    print("global minimum: xmin = {0}, f(xmin) = {1:.6f}".format(ret.x, ret.fun))


