# Supply Chain Model

import numpy as np
import warnings

warnings.filterwarnings("ignore")


class SC_model:
    def __init__(self, num_run, num_period, num_products, prod_demand_var):
        self.num_runs = num_run
        self.num_prd = num_period
        self.num_products = num_products
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
        self.fac_reg_distance = np.array([[32, 21, 19, 31, 22, 35, 18, 33],
                                          [25, 36, 30, 24, 16, 14, 31, 19],
                                          [18, 12, 27, 11, 33, 28, 23, 22]])
        self.max_order = 40
        self.min_order = 1
        self.demand_product_region = np.array([[12, 12, 7, 11, 8, 15, 10, 15],
                                               [10, 13, 11, 5, 8, 13, 11, 8],
                                               [15, 12, 16, 20, 13, 13, 7, 13],
                                               [6, 11, 7, 8, 10, 7, 10, 7],
                                               [18, 14, 18, 13, 18, 13, 16, 18]])

        self.M = 10 ** 4
        self.satisfactory_thresh = [0.9] * self.num_products


    def demands(self):
        demands = np.zeros((self.num_runs, self.num_prd, self.num_products, self.num_regions))
        for i in range(self.num_runs):
            for j in range(self.num_prd):
                for k in range(self.num_products):
                    for c in range(self.num_regions):
                        demands[i,j,k,c] = np.round(np.random.normal(self.demand_product_region[k,c], np.sqrt(self.prod_dmnd_var[k])))
        demands[demands<0]=0

        return np.array(demands)


    def simulation(self, invent_coeffs):
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
            target_inventory[:,j] = np.multiply(target_inventory[:,j], invent_coeffs)

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

                            y_count[i,t,k,c] += 1
                            demands[i,t,k,c] = 0

        pr_region = 0
        for i in range(self.num_runs):
            for j in range(self.num_prd):
                pr_region += y_count[i,j]

        pr = np.sum(pr_region,axis=1)

        customer_dissatis = pr/(self.num_runs*self.num_prd*self.num_regions)


        for i in range(self.num_runs):
            for t in range(self.num_prd):
                produced_cost.append(np.dot(produced_level[i,t,...].sum(axis=1), self.production_cost))
                transported_cost.append(np.dot(transported_level[i,t,...].sum(axis=1), np.repeat(self.transp_cost, self.num_products)))
                outsourced_cost.append(np.dot(outsourced_level[i,t,...].sum(axis=1), self.outsource_cost))
                inventory_cost.append(np.dot(inventory_level[i,t, ...].sum(axis=1), self.inventory_cost))



        mean_cost = (np.array(inventory_cost).sum() + np.array(outsourced_cost).sum() +
                np.array(produced_cost).sum() + np.array(transported_cost).sum())/self.num_runs


        cost_dis = [None]*len(self.satisfactory_thresh)
        cost_model = mean_cost

        for i in range(len(self.satisfactory_thresh)):
            if customer_dissatis[i] >= (1 - self.satisfactory_thresh[i]):
                mean_cost += self.M * customer_dissatis[i]
                cost_dis[i] = self.M * customer_dissatis[i]
            else:
                mean_cost += 0
                cost_dis[i]= 0


        cost_dissat = np.sum(cost_dis)

        ############ Mean_cost is  the supply chain cost and dissatisfaction cost
        ############ cost_dissat is total customer dissatisfaction
        ############ cost_model is the supply chain model cost

        return mean_cost, cost_model, cost_dissat


