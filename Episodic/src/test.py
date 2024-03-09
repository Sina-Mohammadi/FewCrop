import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


from models.standard import __dict__ as standard_dict
from models.meta import __dict__ as meta_dict
from methods import __dict__ as all_methods
from utils import load_cfg_from_cfg_file, merge_cfg_from_list, \
                   get_model_dir, make_episode_visualization, \
                   load_checkpoint, blockPrint, enablePrint, \
                   compute_confidence_interval


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args):

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = os.path.join(args.res_path, args.method)
    os.makedirs(res_path, exist_ok=True)

    # ============ Testing method ================
    method = all_methods[args.method](args=args)
    average_acc = []
    average_std = []

    for seed in args.seeds:
        args.seed = seed
        # ============ Data loaders =========

        from dataset.loader import DatasetFolder
        from torch.utils.data import DataLoader



        from dataset import Tasks_Generator, CategoriesSampler, get_dataset, get_dataloader


        
        test_set = get_dataset(split='test',scenario=args.scenario)
        sampler = CategoriesSampler(label=test_set.labels, n_batch=args.batch_size, n_cls=args.num_ways, s_shot=args.num_support, q_shot=args.num_support, balanced=args.sampling_strategy, alpha=2)

        loader = DataLoader(test_set, sampler=sampler,
                            num_workers=0, pin_memory=True)

        task_generator = Tasks_Generator(n_ways=args.num_ways, shot=args.num_support, loader=loader,
                                     train_mean=0, log_file=1)            

        # ============ Model ================
        model_dir = get_model_dir(args, seed)
        if 'MAML' in args.method:
            print(f"Meta {args.arch} loaded")
            model = meta_dict[args.arch](num_classes=args.num_ways)
        else:
            print(f"Standard {args.arch} loaded")
            model = standard_dict[args.arch](num_classes=args.num_ways)
        
        if args.PretrainedOnSource==True:
            load_checkpoint(model, model_dir, type='best')
        model = model.to(device)

        # ============ Training loop ============
        model.eval()
        method.eval()
        print(f'Starting testing for seed {seed}')
        test_acc = 0.
        test_loss = 0.
        predictions = []
        results = []
        results1 = []
        mIOUs=[]
        IOUs=[]
        mUAs=[]
        UAs=[]
        mPAs=[]
        PAs=[]  
        F1s=[] 
        mF1s=[]
        CMs=[]
        
        tqdm_bar = tqdm(range(args.test_iter), total=args.test_iter, ascii=True)
        i = 0

        for i in tqdm(range(args.test_iter)):
            tasks = task_generator.generate_tasks()

            support_labels, query_labels = tasks['y_s'][...,0], tasks['y_q'][...,0]
            support, query = tasks['x_s'], tasks['x_q']
 

            if args.num_ways==24:
                y_q = query_labels.numpy()
                x_q = query.numpy()
                x_qu=[]
                y_qu=[]
                for i in range(len(y_q)):
                    indexes=np.argsort(y_q[i])
                    x_qu.append(x_q[i][indexes])
                    y_qu.append(y_q[i][indexes])
                query=torch.from_numpy(np.array(x_qu) )
                query_labels=torch.from_numpy(np.array(y_qu) )


            
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)
            

                    
            if args.method == 'MAML':  # MAML uses transductive batchnorm, whic corrupts the model
                blockPrint()
                if args.PretrainedOnSource==True:
                    load_checkpoint(model, model_dir, type='best')
                    
                enablePrint()

            # ============ Evaluation ============
            loss, preds_q = method(x_s=support,
                                        x_q=query,
                                        y_s=support_labels,
                                        y_q=query_labels,
                                        model=model)
            if args.visu and i % 100 == 0:
                task_id = 0
                root = os.path.join(model_dir, 'visu', 'test')
                os.makedirs(root, exist_ok=True)
                save_path = os.path.join(root, f'{i}.png')
                make_episode_visualization(
                           args,
                           support[task_id].cpu().numpy(),
                           query[task_id].cpu().numpy(),
                           support_labels[task_id].cpu().numpy(),
                           query_labels[task_id].cpu().numpy(),
                           preds_q[task_id].cpu().numpy(),
                           save_path)
            predictions.append((preds_q == query_labels).float().mean().item())
            test_acc, test_std = compute_confidence_interval(predictions)
            
            #print(test_acc)
            preds_q_onehot=F.one_hot(preds_q, num_classes=args.num_ways)
            y_q_onehot=F.one_hot(query_labels, num_classes=args.num_ways)
            
            
            IOUs_=[]
            for j in range(query_labels.shape[0]):
                iou=[]
                for i in range(args.num_ways):

                    IOU_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (preds_q_onehot[j,:,i]+y_q_onehot[j,:,i]-(preds_q_onehot[j,:,i]*y_q_onehot[j,:,i])).cpu()) )
                    IOU_i=np.nan_to_num(IOU_i, copy=True, nan=0.0, posinf=None, neginf=None)
                    iou.append(IOU_i)
                IOUs_.append(np.array(iou))
            IOUs_=np.array(IOUs_)

            UAs_=[]
            PAs_=[]
            F1s_=[]
            for j in range(query_labels.shape[0]): 
                ua=[]
                pa=[]
                f1=[]
                for i in range(args.num_ways):
                    UA_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (preds_q_onehot[j,:,i]).cpu())==1 )
                    UA_i=np.nan_to_num(UA_i, copy=True, nan=0.0, posinf=None, neginf=None)
                    ua.append(UA_i)
                    PA_i=np.sum(np.array( (preds_q_onehot[j,:,i]*y_q_onehot[j,:,i]).cpu() ) )/np.sum(  np.array( (y_q_onehot[j,:,i]).cpu())==1 )
                    PA_i=np.nan_to_num(PA_i, copy=True, nan=0.0, posinf=None, neginf=None)
                    pa.append(PA_i)
                    F1_i=2*(UA_i*PA_i)/(UA_i+PA_i)
                    F1_i=np.nan_to_num(F1_i, copy=True, nan=0.0, posinf=None, neginf=None)   
                    f1.append(F1_i)
                UAs_.append(np.array(ua))
                PAs_.append(np.array(pa))
                F1s_.append(np.array(f1))

            UAs_=np.array(UAs_)      
            PAs_=np.array(PAs_)            
            F1s_=np.array(F1s_)            


            IOUs.append(IOUs_) #list containing arrays with size batch_size x num_ways
            mIOU=np.mean(IOUs_,axis=1) # shape: (n_task or batch_size)
            mIOUs.extend(mIOU)
            iou_mean, iou_conf = compute_confidence_interval(mIOUs)
            print("iou_mean:",iou_mean)
            print("iou_conf:",iou_conf) 


            UAs.append(UAs_)
            mUA=np.mean(UAs_,axis=1) # shape: (n_task or batch_size)
            mUAs.extend(mUA)
            ua_mean, ua_conf = compute_confidence_interval(mUAs)
            print("ua_mean:",ua_mean)
            print("ua_conf:",ua_conf)                     



            PAs.append(PAs_)
            #print(logs['pa'][:,:,-1].shape)
            #print(np.mean(np.array(PAs),axis=(0,2))) #printing PA of each class
            mPA=np.mean(PAs_,axis=1) # shape: (n_task or batch_size)
            mPAs.extend(mPA)
            pa_mean, pa_conf = compute_confidence_interval(mPAs)
            print("pa_mean:",pa_mean)
            print("pa_conf:",pa_conf)                      
            
            
            F1s.append(F1s_)
            #print(logs['pa'][:,:,-1].shape)
            #print(np.mean(np.array(PAs),axis=(0,2))) #printing PA of each class
            mF1=np.mean(F1s_,axis=1) # shape: (n_task or batch_size)
            mF1s.extend(mF1)
            f1_mean, f1_conf = compute_confidence_interval(mF1s)
            print("f1_mean:",f1_mean)
            print("f1_conf:",f1_conf)                
            
            

            CM=confusion_matrix(y_true=query_labels.cpu().numpy().flatten(), y_pred=preds_q.cpu().numpy().flatten(),labels=np.arange(args.num_ways))       

            conf_matrix= np.array(CM) #batch_sizexbatch_size               
            
            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_acc_iou_ua_pa.npy", {"acc":test_acc,"acc_std":test_std,"iou":iou_mean,"iou_std":iou_conf,"ua":ua_mean,"ua_std":ua_conf,"pa":pa_mean,"pa_std":pa_conf, "f1":f1_mean,"f1_std":f1_conf} )                   


            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_IOU.npy", np.mean(np.array(IOUs),axis=(0,1)) )                                 



            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_UA.npy", np.mean(np.array(UAs),axis=(0,1)) )                

            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_PA.npy", np.mean(np.array(PAs),axis=(0,1)) )                   


            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_F1.npy", np.mean(np.array(F1s),axis=(0,1)) ) 


            CMs.append(conf_matrix) #List of arrays containing n_ways x n_ways matrices
            np.save(args.method+str(args.num_support)+"shots"+
                    str(args.num_ways)+"ways"+args.sampling_strategy+str(args.PretrainedOnSource)+"_CM.npy",np.sum(np.array(CMs),axis=0) )

          
            
            if loss is not None:
                test_loss += loss.detach().mean().item()
            if i % 10 == 0:
                tqdm_bar.set_description(f'Test Prec@1 {test_acc :.3f}  \
                                           Test 95CI@1 {test_std :.4f}  \
                                           Test loss {test_loss / (i+1):.3f}',
                                         )
                update_csv(args=args,
                           task_id=i,
                           acc=test_acc,
                           std=test_std,
                           path=os.path.join(res_path, 'test.csv'))
            if i >= args.test_iter:
                break
            i += 1
        average_acc.append(test_acc)
        average_std.append(test_std)

    print(f'--- Average accuracy over {len(args.seeds)} seeds = {np.mean(average_acc):.3f}')
    print(f'--- Average 95\% Confidence Interval over {len(args.seeds)} seeds = {np.mean(average_std):.4f}')


def update_csv(args: argparse.Namespace,
               task_id: int,
               acc: float,
               std: float,
               path: str):
    # res = OrderedDict()
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame()
    records = res.to_dict('records')
    # Check whether the entry exist already, if yes, simply update the accuracy
    match = False
    for entry in records:
        match = [str(value) == str(args[param]) for param, value in list(entry.items()) if param not in ['acc', 'task', 'std']]
        match = (sum(match) == len(match))
        if match:
            entry['task'] = task_id
            entry['acc'] = round(acc, 4)
            entry['std'] = round(std, 4)
            break

    # If entry did not exist, just create it
    if not match:
        new_entry = {param: args[param] for param in args.simu_params}
        new_entry['task'] = task_id
        new_entry['acc'] = round(acc, 4)
        new_entry['std'] = round(std, 4)
        records.append(new_entry)
    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
