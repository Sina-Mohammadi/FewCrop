import torch.nn.functional as F
from src.utils import get_mi, get_cond_entropy, get_entropy, get_one_hot, get_metric, Logger, extract_features
from tqdm import tqdm
import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix

class Baseline(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.temp = args.temp
        self.n_ways = args.n_ways        
        
        self.iter = args.iter
        self.lr = float(args.lr_baseline)
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def init_weights(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s,
                         y_q=y_q)
        self.model.train()

        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def record_info(self, new_time, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        """
        logits_q = self.get_logits(query).detach()
        self.preds_q = logits_q.argmax(2)
        self.y_q=y_q        
        q_probs = logits_q.softmax(2)
        self.timestamps.append(new_time)
        self.test_acc.append((self.preds_q == self.y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        
        
        
        preds_q_onehot=F.one_hot(self.preds_q, num_classes=self.n_ways)
        
        y_q_onehot=F.one_hot(self.y_q, num_classes=self.n_ways)
        IOUs=[]
        for j in range(self.y_q.shape[0]):
            iou=[]
            for i in range(self.n_ways):

                IOU_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (preds_q_onehot[j,:,i]+y_q_onehot[j,:,i]-(preds_q_onehot[j,:,i]*y_q_onehot[j,:,i])).cpu()) )
                IOU_i=np.nan_to_num(IOU_i, copy=True, nan=0.0, posinf=None, neginf=None)
                
                iou.append(IOU_i)
            IOUs.append(np.array(iou))
        IOUs=np.array(IOUs)

        UAs=[]
        PAs=[]
        F1s=[]
        for j in range(self.y_q.shape[0]): 
            ua=[]
            pa=[]
            f1=[]
            for i in range(self.n_ways):
                UA_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (preds_q_onehot[j,:,i]).cpu())==1 )
                UA_i=np.nan_to_num(UA_i, copy=True, nan=0.0, posinf=None, neginf=None)
                ua.append(UA_i)
                PA_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (y_q_onehot[j,:,i]).cpu())==1 )
                PA_i=np.nan_to_num(PA_i, copy=True, nan=0.0, posinf=None, neginf=None)
                pa.append(PA_i)
                F1_i=2*(UA_i*PA_i)/(UA_i+PA_i)
                F1_i=np.nan_to_num(F1_i, copy=True, nan=0.0, posinf=None, neginf=None)   
                f1.append(F1_i)
            UAs.append(np.array(ua))
            PAs.append(np.array(pa))
            F1s.append(np.array(f1))
                
        UAs=np.array(UAs)      
        PAs=np.array(PAs)            
        F1s=np.array(F1s)            



        
        CM=confusion_matrix(y_true=self.y_q.cpu().numpy().flatten(), y_pred=self.preds_q.cpu().numpy().flatten(),labels=np.arange(self.n_ways))       
        

        conf_matrix= np.array(CM) #batch_sizexbatch_size       
        
        
        return {'timestamps': self.timestamps,
                'acc': self.test_acc, 'iou':IOUs , 'ua':UAs , 'pa':PAs , 'cm':conf_matrix , 'f1':F1s}

    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the BASELINE inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        # Record info if there's no Baseline iteration
        if self.iter == 0:
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1 - t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
        else:
            self.logger.info(" ==> Executing Baseline adaptation over {} iterations on {} shot tasks...".format(self.iter, shot))

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        for i in tqdm(range(self.iter)):
            logits_s = self.get_logits(support)
            # logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            loss = ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            self.model.eval()

            self.record_info(new_time=t1 - t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
            t0 = time.time()


    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']

        # Extract features
        z_s, z_q = extract_features(model=self.model, support=x_s, query=x_q)

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Init basic prototypes
        self.init_weights(support=support, y_s=y_s, query=query, y_q=y_q)
        # Run adaptation
        self.run_adaptation(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

class Baseline_PlusPlus(Baseline):
    def __init__(self, model, device, log_file, args):
        super().__init__(model=model, device=device, log_file=log_file, args=args)

    def record_info(self, new_time, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)

        preds_q = []
        for j in range(self.number_tasks):
            distance = get_metric('cosine')(self.weights[j], query[j])
            preds_q.append(torch.argmin(distance, dim=1))
        preds_q = torch.stack(preds_q)
        self.preds_q=preds_q
        self.y_q=y_q
        self.timestamps.append(new_time)
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))


    def run_adaptation(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the BASELINE++ inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        # Record info if there's no Baseline iteration
        if self.iter == 0:
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1 - t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
        else:
            self.logger.info(" ==> Executing Baseline++ adaptation over {} iterations on {} shot tasks...".format(self.iter, shot))

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        for i in tqdm(range(self.iter)):
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)
            loss = ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            self.model.eval()

            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
            t0 = time.time()