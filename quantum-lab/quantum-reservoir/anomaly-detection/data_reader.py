import os,sys
from utils import Unpickle


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import print_events


def scale_feature(X,epsilon=0.,max_array=None,min_array=None):
    scaler =  MinMaxScaler(feature_range = (0 , np.pi-epsilon))
    if max_array is None:
        X = scaler.fit_transform(X)
        passed=np.arange(len(X))
    else: 
        assert min_array is not None
        #max_array=[500.,500.]
        print (max_array,np.max(X,axis=0))
        passed_inds=[]
        for array,M in zip(np.swapaxes(X,0,1),max_array):
            passed_inds.append(array<M)
        passed_inds=np.array(passed_inds)
        passed_inds=np.sum(passed_inds,axis=0)
        rejected=np.where(passed_inds != X.shape[-1])[0]
        passed_inds=passed_inds==X.shape[-1]
        X=X[passed_inds]
        #print (X.shape,np.min(X,axis=0),np.max(X,axis=0))
        X=np.concatenate(([min_array],X,[max_array]),axis=0)
        #print (X.shape,X,np.min(X,axis=0),np.max(X,axis=0))
        X=scaler.fit_transform(X)[1:-1]
        #print (X.shape,X[1],X[-1])
        #sys.exit()
        #print (passed_inds.shape,np.max(X,axis=0),X.shape,rejected)
    return X,passed_inds

import __main__   

def get_seperate(**kwargs):
    unscaled=kwargs.get("unscaled",False)
    main_filename=__main__.__file__.split('/')[-1]
    print (main_filename)
    if unscaled: assert main_filename!='auto_qml.py'
    #sys.exit()
    print (kwargs)
    max_values={'lep1pt':1000, 
                'lep2pt':900, 
                'theta_ll':np.pi,
                'b1pt':1000, 
                'b2pt':900,
                'theta_bb':np.pi, 
                'MET':1000
                }
    min_values={'lep1pt':0., 
                'lep2pt':0., 
                'theta_ll':0.,
                'b1pt':0., 
                'b2pt':0.,
                'theta_bb':0., 
                'MET':0.,
                } # MINIMUM OF NON PERIODIC VARIABLES SET TO 0 FOR PRESERVING TOPOLOGY
    combined_signal=kwargs.get('combined_signal',False)
    all_files=['sample_bg_morestat.csv','sample_mH500_morestat.csv',
               'sample_mH1000_morestat.csv','sample_mH1500_morestat.csv',
               'sample_mH2000_morestat.csv','sample_mH2500_morestat.csv']
    #'sample_bg.csv','sample_mH500.csv','sample_mH1000.csv','sample_mH1500.csv','sample_mH2000.csv','sample_mH2500.csv'
    csv_dir = './data/'
    keys=kwargs.get('keys')
    max_array=[]
    min_array=[]
    for item in keys:
        min_array.append(min_values[item])
        max_array.append(max_values[item])
    min_array,max_aray=np.array(min_array),np.array(max_array)
    assert keys is not None,'Provide compulsory kwarg keys=<list of df keys>'
    train=kwargs.get('train',True)
    signal_files=[]
    bg_filename=kwargs.get('train_file')#'sample_bg.csv'
    signal_files=all_files+[]
    assert bg_filename in all_files,f"Provide compulsory kwarg: train_file in\n {all_files}"
    signal_files.remove(bg_filename)
    print (train,signal_files,bg_filename,keys)
    bkg = pd.read_csv(csv_dir+bg_filename,delimiter=',')
    unscaled_bg_array=bkg[keys].values
    print ('Max before scaling:',np.max(unscaled_bg_array,axis=0),np.min(unscaled_bg_array,axis=0))
    bg_array,passed=scale_feature(unscaled_bg_array,max_array=max_array,min_array=min_array)
    unscaled_bg_array=unscaled_bg_array[passed]
    if unscaled:
        bg_array=unscaled_bg_array
    print ('Max after scaling:',np.max(unscaled_bg_array,axis=0),np.min(unscaled_bg_array,axis=0))
    #sys.exit()
    print (np.max(bg_array,axis=0),np.min(bg_array,axis=0))
    if 'bg' in bg_filename:
        train_X=bg_array[:30000]
        val_X=bg_array[30000:35000]
        test_X=bg_array[35000:50000]
    else:
        length=len(bg_array)
        train_X=bg_array[:int(0.5*length)]
        val_X=bg_array[int(0.5*length):int(0.75*length)]
        test_X=bg_array[int(0.75*length):]
    print (train_X.shape,val_X.shape,test_X.shape)
    if train:
        return {
                'train':{'X':train_X,'y':np.zeros(len(train_X))},
                'val':{'X':val_X,'y':np.zeros(len(val_X))},
               }
    return_dict={bg_filename:test_X}
    print (all_files,signal_files)           
    for item in signal_files:
        print ('\nLoading :',item)
        events=pd.read_csv(csv_dir+item,delimiter=',')
        unscaled_values=events[keys].values   
        print ('Original shape:',unscaled_values.shape) 
        print ('Max before scaling:',np.max(unscaled_values,axis=0))
        values,passed=scale_feature(unscaled_values,max_array=max_array,min_array=min_array)
        unscaled_values=unscaled_values[passed]
        print ('Max after scaling:',np.max(unscaled_values,axis=0))  
        print ('Scaled shape:',values.shape,np.min(values,axis=0),np.max(values,axis=0))
        #break
        if unscaled: return_dict[item]=unscaled_values
        else: return_dict[item]=values
    return_dict['train_filename']=bg_filename
    
    if not combined_signal:
        print_events(return_dict,name='test dictionary') 
        return return_dict
    X,y,y_map=[],[],{}
    i=0
    for key,val in return_dict.items():
        if key=='train_filename': continue
        X.append(val)
        y_map[i]=key
        y.append(np.full(len(val),i))
    test_dict={'X':np.concatenate(X,axis=0),'y':np.concatenate(y,axis=0),'y_map':y_map,'train_filename':bg_filename}
    print_events(test_dict,name='test_dict')
    return test_dict    
    sys.exit()
    
def new_get_data(autoencoder=True,scale=True,return_splitted=False,return_dataframe=False,zero_bg=True,return_names=False,return_all=False,
                 **kwargs):
    csv_dir = './data/'
    keys=kwargs.get('keys')
    assert keys is not None,"Provide keys in ['lep1pt', 'lep2pt', 'theta_ll', 'b1pt', 'b2pt','theta_bb', 'MET']!"
    bkg = pd.read_csv(csv_dir+'sample_bg.csv',delimiter=',',nrows=12000) # trained qcompress with nrows=1250
    if zero_bg: bkg['s or b'] = 0.
    else: bkg['s or b'] = -1
    sig = pd.read_csv(csv_dir+'sample_sig.csv',delimiter=',',nrows=12000)
    sig['s or b'] = 1

    all_events = pd.concat((bkg,sig))
    all_events = all_events.reset_index().drop('index',axis=1)

    all_events = all_events[keys+['s or b']]
    #if return_dataframe: return all_events
    X, y = all_events.loc[:, all_events.columns != 's or b'], all_events['s or b']
    if scale:
        scaler =  MinMaxScaler(feature_range = (0 , np.pi))
        X = scaler.fit_transform(X)
        y = y.to_numpy()
    else: X,y=X.to_numpy(),y.to_numpy()
    #if return_all:
    #    if return_names:return X,y,['MET',  'b1 pT']
    #    else: return X,y
        
    #print (X,y)
    #if not return_splitted: return {'X':X,'y':y}
    X_train, X_remain, y_train, y_remain = train_test_split(
     X, y, test_size=0.8, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(
         X_remain, y_remain, test_size=0.5, random_state=42)
    print (X_train.shape,X_test.shape,X_val.shape)
    if autoencoder:
        print ('Returning only background events for training!')
        #print (y_train,y_train.shape)
        inds=y_train!=1
        X_train=X_train[inds]
        y_train=y_train[inds]
        #print (y_train.shape,y_train)
    #sys.exit()
    if kwargs.get('train',False):
        inds=y_val !=1
        X_val=X_val[inds]
        y_val=y_val[inds]
    return {
            'train': {'X':X_train,'y':y_train},
            'val'  : {'X':X_val,'y':y_val},
            'test' : {'X':X_test,'y':y_test},
            }

def old_get_data(autoencoder=True,scale=True,return_splitted=False,return_dataframe=False,zero_bg=True,return_names=False,return_all=False,
                 **kwargs):
    csv_dir = './data/'
    
    bkg = pd.read_csv(csv_dir+'background_small.csv',delimiter=',',nrows=50000) # trained qcompress with nrows=1250
    if zero_bg: bkg['s or b'] = 0.
    else: bkg['s or b'] = -1
    sig = pd.read_csv(csv_dir+'signal_small.csv',delimiter=',',nrows=50000)
    sig['s or b'] = 1

    all_events = pd.concat((bkg,sig))
    all_events = all_events.reset_index().drop('index',axis=1)

    all_events = all_events[['MET',  'b1 pT','s or b']]
    #if return_dataframe: return all_events
    X, y = all_events.loc[:, all_events.columns != 's or b'], all_events['s or b']
    if scale:
        scaler =  MinMaxScaler(feature_range = (0 , np.pi))
        X = scaler.fit_transform(X)
        y = y.to_numpy()
    else: X,y=X.to_numpy(),y.to_numpy()
    #if return_all:
    #    if return_names:return X,y,['MET',  'b1 pT']
    #    else: return X,y
        
    #print (X,y)
    #if not return_splitted: return {'X':X,'y':y}
    X_train, X_remain, y_train, y_remain = train_test_split(
     X, y, test_size=0.8, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(
         X_remain, y_remain, test_size=0.5, random_state=42)
    print (X_train.shape,X_test.shape,X_val.shape)
    if autoencoder:
        print ('Returning only background events for training!')
        #print (y_train,y_train.shape)
        inds=y_train!=1
        X_train=X_train[inds]
        y_train=y_train[inds]
        #print (y_train.shape,y_train)
    #sys.exit()
    if kwargs.get('train',False):
        inds=y_val !=1
        X_val=X_val[inds]
        y_val=y_val[inds]
    
    return {
            'train': {'X':X_train,'y':y_train},
            'val'  : {'X':X_val,'y':y_val},
            'test' : {'X':X_test,'y':y_test},
            }
def parton_data(bg_only=True,scale=True,return_all=False,x_attr=None,return_names=False,**kwargs):
    run_names=['zz_ll_inv','hdm_h_ll_z_inv']
    bg_name=run_names[0]
    sg_name=run_names[1]
    path='./lhc_vars/'
    bkg = pd.read_csv(path+bg_name+'.csv')
    if x_attr is None: x_attr=list(bkg.keys())
    bkg['s or b'] = -1
    #print (bkg.head(),'\n',x_attr)
    if not bg_only:
        sig = pd.read_csv(path+sg_name+'.csv')
        sig['s or b'] = 1

        all_events = pd.concat((bkg,sig))
        
    else: all_events=bkg
    all_events = all_events.reset_index().drop('index',axis=1)

    all_events = all_events[x_attr+['s or b']]
    
    X, y = all_events.loc[:, all_events.columns != 's or b'], all_events['s or b']
    if scale:
        scaler =  MinMaxScaler(feature_range = (0 , np.pi))
        X = scaler.fit_transform(X)
        y = y.to_numpy()
    else: X,y=X.to_numpy(),y.to_numpy()
    #print (X,y)
    if return_all:
        if return_names:return X,y,x_attr
        else: return X,y

    X_train, X_remain, y_train, y_remain = train_test_split(
     X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(
         X_remain, y_remain, test_size=0.5, random_state=42)
    print (X_train.shape,X_test.shape,X_val.shape)
    return {
            'train': {'X':X_train,'y':y_train},
            'val'  : {'X':X_val,'y':y_val},
            'test' : {'X':X_test,'y':y_test},
            }
