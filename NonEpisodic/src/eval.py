import numpy as np
from src.utils import warp_tqdm, compute_confidence_interval, load_checkpoint, Logger, extract_mean_features
from src.methods.tim import ALPHA_TIM, TIM_GD

from src.methods.simpleshot import SimpleShot
from src.methods.baseline import Baseline, Baseline_PlusPlus

from src.methods.entropy_min import Entropy_min
from src.datasets import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader
import torch

class Evaluator:
    def __init__(self, device, args, log_file):
        self.device = device
        self.args = args
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)

    def run_full_evaluation(self, model):
        """
        Run the evaluation over all the tasks
        inputs:
            model : The loaded model containing the feature extractor
            args : All parameters

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        self.logger.info("=> Runnning full evaluation with method: {}".format(self.args.method))
        if self.args.PretrainedOnSource==True:
            load_checkpoint(model=model, model_path=self.args.ckpt_path, type=self.args.model_tag)
        dataset = {}
        loader_info = {'aug': False, 'out_name': False}



        train_set = get_dataset('train', scenario=self.args.scenario)
        dataset['train_loader'] = train_set

        test_set = get_dataset(self.args.used_set, scenario=self.args.scenario)
        dataset.update({'test': test_set})

        train_loader = get_dataloader(sets=train_set, args=self.args)
        train_mean, _ = extract_mean_features(model=model,  train_loader=train_loader, args=self.args,
                                              logger=self.logger, device=self.device)

        results = []
        results1 = []
        mIOUs=[]
        IOUs=[]
        mUAs=[]
        UAs=[]
        mPAs=[]
        PAs=[]
        mF1s=[]
        F1s=[]   
        CMs=[]
        for shot in self.args.shots:
            sampler = CategoriesSampler(dataset['test'].labels, self.args.batch_size,
                                        self.args.n_ways, shot, self.args.n_query,
                                        self.args.sampling_strategy, self.args.alpha_dirichlet)

            test_loader = get_dataloader(sets=dataset['test'], args=self.args,
                                         sampler=sampler, pin_memory=True)
            task_generator = Tasks_Generator(n_ways=self.args.n_ways, shot=shot, loader=test_loader,
                                             train_mean=train_mean, log_file=self.log_file)
            results_task = []
            results_task1 = []
            
            for i in range(int(self.args.number_tasks/self.args.batch_size)):
                print('iter',i)
                method = self.get_method_builder(model=model)
                    
                tasks = task_generator.generate_tasks()
                
                if self.args.n_ways==24:
                    y_q = tasks['y_q'][...,0]
                    x_q = tasks['x_q']
                    x_qu=[]
                    y_qu=[]
                    for i in range(len(y_q)):
                        indexes=np.argsort(y_q[i])
                        x_qu.append(x_q[i][indexes].cpu().numpy())
                        y_qu.append(y_q[i][indexes].cpu().numpy())

                    tasks['x_q']=torch.from_numpy( np.array(x_qu) )
                    tasks['y_q']=torch.from_numpy( np.expand_dims(np.array(y_qu),axis=-1) )
                
                
                
                logs = method.run_task(task_dic=tasks, shot=shot)

                results_task1.extend(logs['acc'][:, -1])
                acc_mean, std=compute_confidence_interval(results_task1)
                print(acc_mean)
                print(std)
                
                IOUs.append(logs['iou']) #list containing arrays with size batch_size x num_ways
                
                mIOU=np.mean(logs['iou'],axis=1) # shape: (n_task or batch_size)
                mIOUs.extend(mIOU)
                iou_mean, iou_conf = compute_confidence_interval(mIOUs)
                print("iou_mean:",iou_mean)
                print("iou_conf:",iou_conf) 
                
                
                UAs.append(logs['ua'])
                
                
                mUA=np.mean(logs['ua'],axis=1) # shape: (n_task or batch_size)
                mUAs.extend(mUA)
                ua_mean, ua_conf = compute_confidence_interval(mUAs)
                print("ua_mean:",ua_mean)
                print("ua_conf:",ua_conf)                     
                

                
                PAs.append(logs['pa'])
                
                
                mPA=np.mean(logs['pa'],axis=1) # shape: (n_task or batch_size)
                mPAs.extend(mPA)
                pa_mean, pa_conf = compute_confidence_interval(mPAs)
                print("pa_mean:",pa_mean)
                print("pa_conf:",pa_conf)                
                
                F1s.append(logs['f1'])
                
                
                mF1=np.mean(logs['f1'],axis=1) # shape: (n_task or batch_size)
                mF1s.extend(mF1)
                f1_mean, f1_conf = compute_confidence_interval(mF1s)
                print("f1_mean:",f1_mean)
                print("f1_conf:",f1_conf)   
                
                
                
                
                
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_acc_iou_ua_pa_f1.npy", {"acc":acc_mean,"acc_std":std,"iou":iou_mean,"iou_std":iou_conf,"ua":ua_mean,"ua_std":ua_conf,"pa":pa_mean,"pa_std":pa_conf,"f1":f1_mean,"f1_std":f1_conf} )                   
                
                
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_IOU.npy", np.mean(np.array(IOUs),axis=(0,1)) )                                 
                
                
                
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_UA.npy", np.mean(np.array(UAs),axis=(0,1)) )                
                
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_PA.npy", np.mean(np.array(PAs),axis=(0,1)) )                   
                
                
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_F1.npy", np.mean(np.array(F1s),axis=(0,1)) ) 
                
                
                CMs.append(logs['cm']) #List of arrays containing n_ways x n_ways matrices
                np.save(self.args.scenario+'-'+self.args.method+str(self.args.shots[0])+"shots"+
                        str(self.args.n_ways)+"ways"+self.args.sampling_strategy+str(self.args.PretrainedOnSource)+"_CM.npy",np.sum(np.array(CMs),axis=0) )
                
                self.logger.info(" ==> acc {} std {} iou {} iou_std {} ua {} ua_std {} pa {} pa_std {}".format(acc_mean, std, iou_mean, iou_conf , ua_mean, ua_conf ,pa_mean, pa_conf))
                del method
            results.append(results_task)
            

        mean_accuracies = np.asarray(results).mean(1)
        mean_conf = std
        
        self.logger.info('----- Final test results -----')
        for shot in self.args.shots:
            self.logger.info('{}-shot mean test accuracy over {} tasks: {}'.format(shot, self.args.number_tasks,mean_accuracies[self.args.shots.index(shot)]))
            
        print(mean_conf)
        return mean_accuracies,mean_conf

    def get_method_builder(self, model):
        # Initialize method classifier builder
        method_info = {'model': model, 'device': self.device, 'log_file': self.log_file, 'args': self.args}
        if self.args.method == 'ALPHA-TIM':
            method_builder = ALPHA_TIM(**method_info)
        elif self.args.method == 'TIM-GD':
            method_builder = TIM_GD(**method_info)
        elif self.args.method == 'LaplacianShot':
            method_builder = LaplacianShot(**method_info)
        elif self.args.method == 'BDCSPN':
            method_builder = BDCSPN(**method_info)
        elif self.args.method == 'SimpleShot':
            method_builder = SimpleShot(**method_info)
        elif self.args.method == 'Baseline':
            method_builder = Baseline(**method_info)
        elif self.args.method == 'Baseline++':
            method_builder = Baseline_PlusPlus(**method_info)
        elif self.args.method == 'PT-MAP':
            method_builder = PT_MAP(**method_info)
        elif self.args.method == 'ProtoNet':
            method_builder = ProtoNet(**method_info)
        elif self.args.method == 'Entropy-min':
            method_builder = Entropy_min(**method_info)
        else:
            self.logger.exception("Method must be in ['TIM_GD', 'ALPHA_TIM', 'SimpleShot', 'Baseline', 'Entropy_min']")
            raise ValueError("Method must be in ['TIM_GD', 'ALPHA_TIM', 'SimpleShot', 'Baseline', 'Entropy_min']")
        return method_builder
