import numpy as np 
import random
import pandas as pd 
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib

from IntOpt.Interior.intopt_energy_mlp import MultilayerRegression

budget = 20
test_batchsize = 31
scaler = joblib.load( 'price_scaler.bin')
def inv_scaler_transform(value):
    return scaler.inverse_transform([[ value]]).squeeze()
def actual_obj(prop_data, n_items = 62):
    n = len(prop_data)
    price = prop_data['Actual sales prices'].values
    cst = prop_data[ 'Actual construction costs'].values
    n_instances = n//n_items
    obj_list = []
    for i in range(n_instances):
        p = price[(n_items*i):((i+1)*n_items)]
        c = cst[(n_items*i):((i+1)*n_items)]
        # print("Total cost",sum(c), "Total Profit",sum(p))

        model = gp.Model()
        model.setParam('OutputFlag', 0)
        x = model.addMVar(shape= n_items, lb=0.0, ub=1.0,vtype=gp.GRB.BINARY, name="x")
        model.addConstr(c @ x == n_items*budget, name="eq")
        model.setObjective(p@x, gp.GRB.MAXIMIZE)
        model.optimize()
        sol = x.X
        obj_list.append(inv_scaler_transform(np.dot(sol,p)))
    return np.array(obj_list)



class PriceNet(nn.Module):
    def __init__(self,n_ts_features,n_cat,n_features,lookback,
                 embedding_size,num_layers,hidden_size,target_size=1):
        super().__init__()
        self.n_ts_features = n_ts_features
        self.n_cat = n_cat
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.lookback = lookback

        self.embeddings = nn.Embedding(n_cat, embedding_size) 

        self.lstm = nn.LSTM(n_ts_features, hidden_size, num_layers, 
            batch_first=True)
        # print("Pricenet: embedding size {} n_features {}".format(embedding_size, n_features))
        # print(n_features+ embedding_size)
        self.fc = nn.Linear(n_features+ embedding_size + self.lookback * self.hidden_size,
            target_size)

    def forward(self,x_features,x_cat,x_ts,h):
        x_emb = self.embeddings(x_cat)
        out, h = self.lstm(x_ts, h)
        
        out = torch.flatten(out,start_dim=1)
        x = torch.cat([x_emb, x_features, out], 1)
        # print(x.shape)
        pred = self.fc(x).squeeze()
        return pred, h


class MyCustomDataset():
    def __init__(self, economic_data,properties_data,lookback=5):
        self.x_features=  properties_data.iloc[:,6:13].values.astype(np.float32)
        self.x_cat = properties_data.iloc[:,5].cat.codes.values.astype(np.int64)
        self.y = properties_data.iloc[:,13].values.astype(np.float32)
        self.cost = properties_data.iloc[:,14].values.astype(np.float32)

        self.ts_features = economic_data.iloc[:,2:].values.astype(np.float32)
        self.x_ts = np.zeros((len(self.ts_features),
            lookback,self.ts_features.shape[1])).astype(np.float32)

        
        for i in range(0,len(self.x_ts),lookback):
            self.x_ts[i] = self.ts_features[i:(i+lookback),:]
        self.x_ts = self.x_ts.reshape(-1,lookback,self.ts_features.shape[1])

        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x_features[idx],self.x_cat[idx],self.x_ts[idx], self.y[idx],self.cost[idx]

class two_stage:
    def __init__(self,n_ts_features=19,n_cat=20,n_features=7,lookback=5,
                 embedding_size=5,num_layers=2,hidden_size=10,target_size=1,
                 epochs=8,optimizer=optim.Adam,batch_size=32,**hyperparams):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.n_features = n_features
        self.n_ts_features = n_ts_features
        self.n_cat = n_cat
        self.lookback = lookback


        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs= epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

        self.model = PriceNet(embedding_size=embedding_size,n_features=n_features,
            n_ts_features = n_ts_features,n_cat= n_cat,lookback = lookback,
                        num_layers=num_layers,hidden_size=hidden_size,target_size=target_size)
        
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)
    def fit(self,economic_data,properties_data):
        logging.info("2stage")
        train_df = MyCustomDataset(economic_data,properties_data)
        criterion = nn.L1Loss(reduction='mean') #nn.MSELoss(reduction='mean')
        for e in range(self.epochs):
            logging.info("EPOCH Starts")
            total_loss = 0
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size,shuffle=True)
            for x_f,x_c,x_t,y,cst in train_dl:
                # print("training shape",x_t.shape)
                self.optimizer.zero_grad()
                h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                
                op,states = self.model(x_f,x_c,x_t,(h,c))
                h,c = states
                loss = criterion(op, y)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            logging.info("EPOCH Ends")

            # print("Epoch{} ::loss {}".format(e,total_loss))
    def val_loss(self,economic_data,properties_data):
        test_obj = actual_obj(properties_data,n_items = test_batchsize)


        self.model.eval()
        criterion = nn.L1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(economic_data,properties_data)
        valid_dl = data_utils.DataLoader(valid_df, batch_size= test_batchsize,shuffle=False)
        prediction_loss = 0
        obj_list = []
        for x_f,x_c,x_t,y,cst in valid_dl:
            
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
            loss = criterion(op, y)
            prediction_loss += loss.item()

            model = gp.Model()
            model.setParam('OutputFlag', 0)
            x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.BINARY, name="x")
            model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*budget, name="eq")
            model.setObjective((op.detach().numpy())@x, gp.GRB.MAXIMIZE)
            model.optimize()
            sol = x.X
            y_np = y.detach().numpy()
            obj_list.append(inv_scaler_transform(np.dot(sol,y_np)))

        self.model.train()
        return prediction_loss, test_obj- np.array(obj_list)
    def predict(self,economic_data,properties_data):
        self.model.eval()
        pred_df = MyCustomDataset(economic_data,properties_data)
        pred_dl = data_utils.DataLoader(pred_df, batch_size=self.batch_size,shuffle=False)
        target =[]
        pred = []
        for x_f,x_c,x_t,y,cst in pred_dl:
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
   
            target = target+y.tolist()
            pred= pred+op.squeeze().tolist()
        self.model.train()
        
        return {'prediction':pred,'groundtuth':target}

class SPO:
    def __init__(self,n_ts_features=19,n_cat=20,n_features=7,lookback=5,
                 embedding_size=5,num_layers=2,hidden_size=10,target_size=1,
                 epochs=8,optimizer=optim.Adam,batch_size=32,budget= budget,**hyperparams):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.n_features = n_features
        self.n_ts_features = n_ts_features
        self.n_cat = n_cat
        self.lookback = lookback
        self.budget = budget


        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs= epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

        self.model = PriceNet(embedding_size=embedding_size,n_features=n_features,
            n_ts_features = n_ts_features,n_cat= n_cat,lookback = lookback,
                        num_layers=num_layers,hidden_size=hidden_size,target_size=target_size)
        
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)
    def fit(self,economic_data,properties_data):
        logging.info("SPO")
        train_df = MyCustomDataset(economic_data,properties_data)
        criterion = nn.L1Loss(reduction='mean') #nn.MSELoss(reduction='mean')
        for e in range(self.epochs):
            logging.info("EPOCH Starts")
            # total_loss = 0
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size,shuffle=True)
            for x_f,x_c,x_t,y ,cst in train_dl:
                # print("shape",x_t.shape)
                self.optimizer.zero_grad()
                h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                
                op,states = self.model(x_f,x_c,x_t,(h,c))
                h,c = states
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.CONTINUOUS, name="x")
                model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*self.budget, name="eq")
                model.setObjective((y.detach().numpy())@x, gp.GRB.MAXIMIZE)
                model.optimize()
                x_actual = x.X

                c_spo = (2*op - y)
                # print("SHape",y.shape, c_spo.shape, op.shape)

                model = gp.Model()
                model.setParam('OutputFlag', 0)
                x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.CONTINUOUS, name="x")
                model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*self.budget, name="eq")
                model.setObjective((c_spo.detach().numpy())@x, gp.GRB.MAXIMIZE)
                model.optimize()
                x_SPO = x.X

                grad = torch.from_numpy( x_SPO -x_actual).float()

                op.backward(gradient=grad)
                self.optimizer.step()
            logging.info("EPOCH Ends")
            # print("Epoch{} ::loss {}".format(e,total_loss))

    def val_loss(self,economic_data,properties_data):
        test_obj = actual_obj(properties_data,n_items = test_batchsize)


        self.model.eval()
        criterion = nn.L1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(economic_data,properties_data)
        valid_dl = data_utils.DataLoader(valid_df, batch_size= test_batchsize,shuffle=False)
        prediction_loss = 0
        obj_list = []
        for x_f,x_c,x_t,y,cst in valid_dl:
            
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
            loss = criterion(op, y)
            prediction_loss += loss.item()

            model = gp.Model()
            model.setParam('OutputFlag', 0)
            x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.BINARY, name="x")
            model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*budget, name="eq")
            model.setObjective((op.detach().numpy())@x, gp.GRB.MAXIMIZE)
            model.optimize()
            sol = x.X
            y_np = y.detach().numpy()
            obj_list.append(inv_scaler_transform(np.dot(sol,y_np)))

        self.model.train()
        return prediction_loss, test_obj- np.array(obj_list)
    def predict(self,economic_data,properties_data):
        self.model.eval()
        pred_df = MyCustomDataset(economic_data,properties_data)
        pred_dl = data_utils.DataLoader(pred_df, batch_size=self.batch_size,shuffle=False)
        target =[]
        pred = []
        for x_f,x_c,x_t,y,cst in pred_dl:
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
   
            target = target+y.tolist()
            pred= pred+op.squeeze().tolist()
        self.model.train()
        
        return {'prediction':pred,'groundtuth':target}

import sys
sys.path.insert(0,"../../Interior")
from ip_model_whole import IPOfunc
class Intopt:
    def __init__(self, param,
                 input_size, hidden_size, num_layers, target_size=1,
                 doScale=True, n_items=48, epochs=1, batchsize=24,
                 verbose=False, validation_relax=True,
                 optimizer=optim.Adam, model_save=False, model_name=None,
                 problem_timelimit=50, model=None, store_validation=False,
                 method=1, mu0=None, smoothing=False, thr=None, max_iter=None,
                 damping=1e-3, clip=0.1, warmstart=False,
                 **hyperparams):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        # self.test_relax = test_relax
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.smoothing = smoothing
        self.thr = thr
        self.damping = damping
        self.hyperparams = hyperparams
        self.max_iter = max_iter
        self.warmstart = warmstart
        self.method = method
        self.mu0 = mu0
        self.clip = clip
        self.problem_timelimit = problem_timelimit
        self.model = model
        self.store_validation = store_validation

        self.model = MultilayerRegression(input_size=input_size,
                                          hidden_size=hidden_size, target_size=target_size, num_layers=num_layers)
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit_icon(self, X, y, X_validation=None, y_validation=None, X_test=None, y_test=None):
        self.model_time = 0.
        runtime = 0.

        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided

        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()

            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation - start_validation

        if test:
            start_test = time.time()

            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time += end_test - start_test

        validation_relax = self.validation_relax
        n_items = self.n_items
        epochs = self.epochs
        batchsize = self.batchsize
        n_batches = X.shape[0] // (batchsize * n_items)
        n_knapsacks = X.shape[0] // n_items
        subepoch = 0

        validation_result = []
        shuffled_batches = [i for i in range(n_batches)]

        max_iter = self.max_iter
        # init_params = {el:None for el in range(n_knapsacks)}

        A, b, G, h, F = make_matrix_intopt(**param)
        logging.info("Started Intopt Optimization with method {} threshold {}".format(self.method, self.thr))

        for e in range(epochs):
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()
                self.optimizer.zero_grad()
                batch_list = random.sample([j for j in range(batchsize)], batchsize)
                for j in batch_list:
                    n_start = (batchsize * shuffled_batches[i] + j) * n_items
                    n_stop = n_start + n_items
                    z = torch.tensor(y[n_start:n_stop], dtype=torch.float)
                    X_tensor = torch.tensor(X[n_start:n_stop, :], dtype=torch.float)
                    c_true = torch.mm(F, torch.tensor(y[n_start:n_stop], dtype=torch.float).unsqueeze(1)).squeeze()

                    c_pred = torch.mm(F, self.model(X_tensor)).squeeze()
                    logging.info("c shape {}".format(c_pred.shape))

                    try:
                        with time_limit(self.problem_timelimit):
                            # print('before IPOfunc')
                            x = IPOfunc(A, b, G, h, pc=True, max_iter=self.max_iter, bounds=[(0., None)],
                                        # init_val= init_params[(batchsize*shuffled_batches[i] + j)],
                                        smoothing=self.smoothing, thr=self.thr, method=self.method,
                                        mu0=self.mu0, damping=self.damping)(c_pred)
                            # print('after IPOfunc')
                            loss = (x * c_true).mean()
                            c_pred.retain_grad()
                            # print('before bacjward')
                            loss.backward()
                            # print('after backward')
                            # torch.nn.utils.clip_grad_norm_(self.lstm_layer.parameters(),
                            #     self.clip)
                        forward_solved = IPOfunc.forward_solved()
                        self.model_time += IPOfunc.Runtime()
                        # print("solving cplt",datetime.datetime.now())
                        # print("solved",sum(x),x.shape)
                    except TimeoutException as msg:
                        forward_solved = False
                        logging.info("timelimitlimit exceeded")
                        print("Epoch[{}::{}] timelimitlimit exceeded\
                           If you see if often consider increasing \
                               problem_timelimit".format(e + 1, i + 1))
                    except LinAlgError as msg:
                        raise
                    except Exception as msg:
                        forward_solved = False
                        logging.info(msg)

                    if forward_solved:
                        logging.info("backward done {} {} {}".format(e, i, j))
                    else:
                        print("Epoch[{}/{}] fwd pass not solved".format(e + 1, i + 1))

                self.optimizer.step()
                end = time.time()
                runtime += end - start
                logging.info("step done {} {}".format(e, i))
                # logging.info("--Model parameters--")
                # for  modelparam in self.lstm_layer.parameters():
                #     logging.info(modelparam)
                # logging.info("--******--")
                if forward_solved:
                    logging.info("fwd not solved")
                    # if any(torch.isnan(c_pred.grad).tolist()):
                    #     logging.info("nan in c-gradient")
                    #     logging.info("smoothing is %s"%self.smoothing)

                subepoch += 1
                print('Epoch[{}/{}], loss(train):{:.2f} @ {:%Y-%m-%d %H:%M:%S} '.format(e + 1,
                                                                                        i + 1, loss.item(),
                                                                                        datetime.datetime.now()))

            if self.model_save:
                torch.save(self.model.state_dict(),
                           str(self.model_name + "_Epoch" + str(e) + "_" + str(i) + ".pth"))

            if self.store_validation:
                y_pred_validation, y_pred_test = return_dict_validation(self, self.model, self.predict, self.param,
                                                                        self.n_items,
                                                                        self.model_save, validation, test,
                                                                        self.model_name, self.validation_relax,
                                                                        X_validation, y_validation, X_test, y_test)
                dict_validation = validation_module(param=param, n_items=self.n_items,
                                                    run_time=runtime, epoch=e, batch=i,
                                                    model_time=self.model_time,
                                                    y_target_validation=y_validation,
                                                    sol_target_validation=self.sol_validation,
                                                    y_pred_validation=y_pred_validation,
                                                    y_target_test=y_test, sol_target_test=self.sol_test,
                                                    y_pred_test=y_pred_test, validation_relax=self.validation_relax)
                validation_result.append(dict_validation)
        if self.store_validation:
            # return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            # self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' % str(datetime.datetime.now()))

            return df

    def validation_result(self, X_validation, y_validation, scaler=None, doScale=True):
        scaler_ = self.scaler if scaler is None else scaler
        return validation_func(X_validation, y_validation, self.param, self.n_items, self.model, scaler_, doScale)

    def predict(self, X, scaler=None, doScale=True):
        scaler_ = self.scaler if scaler is None else scaler
        return predict_func(X, self.model, scaler_, doScale)


    def fit(self,economic_data,properties_data):
        logging.info("Intopt")
        train_df = MyCustomDataset(economic_data,properties_data)

        criterion = nn.L1Loss(reduction='mean') #nn.MSELoss(reduction='mean')
        grad_list = []
        for e in range(self.epochs):

            total_loss = 0
            # for i in range(30):
            #     logging.info("EPOCH Starts")

            #     train_prop = properties_data.sample(n = 279,random_state =i)
            #     valid_prop = properties_data.loc[~properties_data.index.isin(train_prop.index)]
            #     train_sl =  train_prop.Sl.unique().tolist()
            #     valid_sl =  valid_prop.Sl.unique().tolist()
            #     train_prop = train_prop.sort_values(['Sl'],ascending=[True])
            #     valid_prop = valid_prop.sort_values(['Sl'],ascending=[True])

            #     train_econ = economic_data[economic_data.Sl.isin(train_sl)]
            #     valid_econ = economic_data[economic_data.Sl.isin(valid_sl)]
            #     train_econ = train_econ.sort_values(['Sl','Lag'],ascending=[True,False])
            #     valid_econ = valid_econ.sort_values(['Sl','Lag'],ascending=[True,False])
            #     train_df = MyCustomDataset(train_econ,train_prop)

                
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size,shuffle=False)
            
            for x_f,x_c,x_t,y ,cst in train_dl:
                self.optimizer.zero_grad()
                h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                
                op,states = self.model(x_f,x_c,x_t,(h,c))
                h,c = states


                G =  cst.unsqueeze(0)
                h = torch.tensor([x_t.shape[0]*self.budget],dtype=torch.float)


                A = torch.Tensor()
                b = torch.Tensor()


                x = IPOfunc(G,h,A,b,bounds= [(0., 1.)],max_iter=self.max_iter, thr=self.thr,damping=self.damping,
                        smoothing=self.smoothing)(-op)
                loss = -(x*y).mean()
                # op.retain_grad()

                loss.backward()
                self.optimizer.step()

            logging.info("EPOCH Ends")
            # print("Epoch{} ::loss {} ->".format(e,total_loss))
            # print(self.val_loss(valid_econ, valid_prop))
            # print("______________")
    def val_loss(self,economic_data,properties_data):
        test_obj = actual_obj(properties_data,n_items = test_batchsize)


        self.model.eval()
        criterion = nn.L1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(economic_data,properties_data)
        valid_dl = data_utils.DataLoader(valid_df, batch_size= test_batchsize,shuffle=False)
        prediction_loss = 0
        obj_list = []
        for x_f,x_c,x_t,y,cst in valid_dl:
            
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
            loss = criterion(op, y)
            prediction_loss += loss.item()

            model = gp.Model()
            model.setParam('OutputFlag', 0)
            x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.BINARY, name="x")
            model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*budget, name="eq")
            model.setObjective((op.detach().numpy())@x, gp.GRB.MAXIMIZE)
            model.optimize()
            sol = x.X
            y_np = y.detach().numpy()
            obj_list.append(inv_scaler_transform(np.dot(sol,y_np)))

        self.model.train()
        return prediction_loss, test_obj- np.array(obj_list)
    def predict(self,economic_data,properties_data):
        self.model.eval()
        pred_df = MyCustomDataset(economic_data,properties_data)
        pred_dl = data_utils.DataLoader(pred_df, batch_size=self.batch_size,shuffle=False)
        target =[]
        pred = []
        for x_f,x_c,x_t,y,cst in pred_dl:
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
   
            target = target+y.tolist()
            pred= pred+op.squeeze().tolist()
        self.model.train()
        
        return {'prediction':pred,'groundtuth':target}


from qpthlocal.qp import QPFunction
from qpthlocal.qp import QPSolvers
from qpthlocal.qp import make_gurobi_model

def make_matrix_qp(w,budget):
    n = len(w)
    A1 = np.eye(n)
    b1 = np.ones(n)
    A2 = -np.eye(n)
    b2 = np.zeros(n)
    A3 = np.expand_dims(w, axis=0)
    b3 = np.array([budget])

    A = np.vstack([A1,A2])
    b = np.concatenate([b1,b2])

    return torch.from_numpy(A).float(), torch.from_numpy(b).float(),torch.from_numpy(A3).float(), torch.from_numpy(b3).float()
class qptl:
    def __init__(self,tau=1e5,n_ts_features=19,n_cat=20,n_features=7,lookback=5,
                 embedding_size=5,num_layers=2,hidden_size=10,target_size=1,
                 epochs=8,optimizer=optim.Adam,batch_size=32,budget=budget,**hyperparams):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.n_features = n_features
        self.n_ts_features = n_ts_features
        self.n_cat = n_cat
        self.lookback = lookback
        self.budget = budget

        self.tau = tau


        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs= epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

        self.model = PriceNet(embedding_size=embedding_size,n_features=n_features,
            n_ts_features = n_ts_features,n_cat= n_cat,lookback = lookback,
                        num_layers=num_layers,hidden_size=hidden_size,target_size=target_size)
        
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)
    def fit(self,economic_data,properties_data):
        logging.info("QPTL")
        train_df = MyCustomDataset(economic_data,properties_data)
        grad_list = []

        for e in range(self.epochs):
            total_loss = 0
            # for i in range(30):
            #     logging.info("EPOCH Starts")

            #     train_prop = properties_data.sample(n = 279,random_state =i)
            #     valid_prop = properties_data.loc[~properties_data.index.isin(train_prop.index)]
            #     train_sl =  train_prop.Sl.unique().tolist()
            #     valid_sl =  valid_prop.Sl.unique().tolist()
            #     train_prop = train_prop.sort_values(['Sl'],ascending=[True])
            #     valid_prop = valid_prop.sort_values(['Sl'],ascending=[True])

            #     train_econ = economic_data[economic_data.Sl.isin(train_sl)]
            #     valid_econ = economic_data[economic_data.Sl.isin(valid_sl)]
            #     train_econ = train_econ.sort_values(['Sl','Lag'],ascending=[True,False])
            #     valid_econ = valid_econ.sort_values(['Sl','Lag'],ascending=[True,False])
            #     train_df = MyCustomDataset(train_econ,train_prop)



            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size,shuffle=False)
            for x_f,x_c,x_t,y ,cst in train_dl:
                self.optimizer.zero_grad()
                h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
                
                op,states = self.model(x_f,x_c,x_t,(h,c))
                h,c = states
                G,h,A,b = make_matrix_qp(cst.detach().numpy(),x_t.shape[0]*self.budget)
                Q = torch.eye(x_t.shape[0])/self.tau

                model_params_quad = make_gurobi_model(G.detach().numpy(),h.detach().numpy(),
                    A.detach().numpy(),b.detach().numpy(), self.tau*np.eye(x_t.shape[0]))

                x = QPFunction(verbose=False, solver=QPSolvers.GUROBI, 
                    model_params=model_params_quad)(Q.expand(1, *Q.shape), 
                    -op, G.expand(1, *G.shape), h.expand(1, *h.shape),
                    A.expand(1, *A.shape), b.expand(1, *b.shape))
                loss =  -(x*y).mean()
                op.retain_grad()

                loss.backward()
                # op_grad = copy.deepcopy(op.grad)
                # grad_dict = {}
                # grad_dict['epoch'] = e
                # grad_dict['subepoch'] = i
                # for l in range(len(op_grad)):
                #     grad_dict['qpt_cgrad'] = op_grad[l].item()
                #     grad_dict['prediction'] = op[l].item()
                #     grad_dict['true'] = y[l].item() 
                #     grad_list.append(copy.deepcopy(grad_dict))
                self.optimizer.step()
                total_loss += loss.item()
            logging.info("EPOCH Ends")

            # print("Epoch{} ::loss {} ->".format(e,total_loss))
            # print(self.val_loss(valid_econ, valid_prop))
            # print("______________")


    def val_loss(self,economic_data,properties_data):
        test_obj = actual_obj(properties_data,n_items = test_batchsize)


        self.model.eval()
        criterion = nn.L1Loss(reduction='sum')#nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(economic_data,properties_data)
        valid_dl = data_utils.DataLoader(valid_df, batch_size= test_batchsize,shuffle=False)
        prediction_loss = 0
        obj_list = []
        for x_f,x_c,x_t,y,cst in valid_dl:
            
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
            loss = criterion(op, y)
            prediction_loss += loss.item()

            model = gp.Model()
            model.setParam('OutputFlag', 0)
            x = model.addMVar(shape= x_t.shape[0], lb=0.0, ub=1.0,vtype=gp.GRB.BINARY, name="x")
            model.addConstr(cst.detach().numpy() @ x == x_t.shape[0]*budget, name="eq")
            model.setObjective((op.detach().numpy())@x, gp.GRB.MAXIMIZE)
            model.optimize()
            sol = x.X
            y_np = y.detach().numpy()
            obj_list.append(inv_scaler_transform(np.dot(sol,y_np)))

        self.model.train()
        return prediction_loss, test_obj- np.array(obj_list)
    def predict(self,economic_data,properties_data):
        self.model.eval()
        pred_df = MyCustomDataset(economic_data,properties_data)
        pred_dl = data_utils.DataLoader(pred_df, batch_size=self.batch_size,shuffle=False)
        target =[]
        pred = []
        for x_f,x_c,x_t,y,cst in pred_dl:
            h =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)
            c =  torch.zeros((self.num_layers,x_t.shape[0],self.hidden_size),dtype=torch.float)

            op,states = self.model(x_f,x_c,x_t,(h,c))
            h,c = states
   
            target = target+y.tolist()
            pred= pred+op.squeeze().tolist()
        self.model.train()
        
        return {'prediction':pred,'groundtuth':target}