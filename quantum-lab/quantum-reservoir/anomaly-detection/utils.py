import os
import multiprocessing
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
pwd= os.getcwd()



def draw_graph(node_inds,edges,ind=0):
    G=nx.Graph()
    edges=list(edges)
    edges.sort()
    G.add_nodes_from(node_inds)
    G.add_edges_from(edges)
    p=Plotter(set_range=False,x_name=None,y_name=None)
    nx.draw(G,ax=p.axes,with_labels=True)
    p.save_fig('graph'+str(ind))   

def choose_frac(nfeatures,efeatures,size=0.25):
    assert len(nfeatures)==len(efeatures)
    l=len(nfeatures)
    indices=np.arange(l)
    if type(size)==float: size=int(size*l)
    np.random.shuffle(indices)
    choosen=indices[:size]
    choosen=np.sort(choosen)
    print (choosen[:10])
    new_features,new_edges=[],[]
    count=0
    for i,j in zip(nfeatures,efeatures):
        if count in choosen:
            new_features.append(i),new_edges.append(j)
            assert i.shape==new_features[-1].shape
            assert j.shape==new_edges[-1].shape,repr(j.shape)+repr(new_edges[-1].shape)
        count+=1
    new_features,new_edges=np.array(new_features),np.array(new_edges)
        
    #sys.exit()
    return new_features,new_edges
class Plotter:
    '''
    '''
    def __init__(self,x_name="$\eta$",y_name="$\phi$",range=None,size=(10,10),projection=None,title=None,set_range=True):
        
        self.image_range={"x":(-5,5),"y":(-5,5)}
        if range !="image" and not range: range=self.image_range
        self.title=title
        self.projection=projection
        if projection != "subplots":
            fig,axes=plt.figure(figsize=size),plt.axes()
            if self.title is not None:fig.suptitle(title)
            if projection=="3d": axes=fig.add_subplot(111, projection="3d") 
            elif projection=="image":pass
            else: fig.add_subplot(axes)
            if projection ==None and range != None and set_range:
                axes.set_xlim(range["x"]),axes.set_ylim(range["y"])
            if projection != "image": axes.set_xlabel(x_name,fontsize=40),axes.set_ylabel(y_name,fontsize=40)
            self.fig,self.axes=fig,axes
        else:self.fig,self.axes=None,None
        self.marker,self.cmap,self.markersize="s","Blues",10
    def save_fig(self,title,extension="png",dpi=100,save_path="plots",**kwargs):
        pwd=os.getcwd()
        if save_path=="plots":
            dirs = os.listdir(os.getcwd())
            if "plots" not in dirs:
                os.mkdir("plots")
            os.chdir(pwd+'/plots')
        else:
            os.chdir(save_path) 
        self.fig.savefig(title+'.'+extension, format=extension, dpi=dpi,bbox_inches=kwargs.get("bbox_inches","tight"),pad_inches=kwargs.get('pad',0.4))
        #plt.show(block=False)
        plt.close()
        print ("Output plot saved at ",os.getcwd(),title+"."+extension)
        os.chdir(pwd)
        return 






def check_dir(path):
    '''check if <path> to dir exists or not. If it doesn't, create the <dir> returns the absolute path to the created dir'''
    pwd=os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
    path=os.getcwd()
    os.chdir(pwd)
    return path
    
def print_events(events,name=None):
    '''Function for printing nested dictionary with at most 3 levels, with final value being a numpy.ndarry, prints the shape of the array'''
    if name: print (name)
    for channel in events:
        if type(events[channel]) == np.ndarray or type(events[channel]) == list:
            if type(events[channel]) == np.ndarray or channel=='EventAttribute': print ("    Final State:", channel,np.array(events[channel]).shape)
            else: 
                try: print ("    Final State:", channel,[item.shape for item in events[channel]])
                except AttributeError:
                    print ("    Final State:", channel,len(events[channel]))
            continue
        print ("Channel: ",channel)
        if type(events[channel]) != dict: continue
        for topology in events[channel]:
            if type(events[channel][topology])!= dict: continue
            if type(events[channel][topology])==np.ndarray  or type(events[channel]) == list:
                print ("    Final State: ",topology, np.array(events[channel][topology]).shape)
                continue
            print ("Topology: ",topology)
            for final_state in events[channel][topology]:
                print ("    Final State: ",final_state," Shape: ",events[channel][topology][final_state].shape)
    return





def Unpickle(filename,path=None,load_path=".",verbose=True,keys=None,extension='.pickle'):
    '''load <python_object> from <filename> at location <load_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: load_path=path
    pwd=os.getcwd()
    if load_path != ".": os.chdir(load_path)
    if filename[-4:]==".npy":
        ret=np.load(filename,allow_pickle=True)
        if verbose: print (filename," loaded from ",os.getcwd())
        os.chdir(pwd)
        return ret
    try:
        with open(filename,"rb") as File:
            return_object=pickle.load(File)
    except Exception as e:
        print (e," checking if folder with ",filename.split(".")[0]," exists..")
        try: os.chdir(filename.split(".")[0])   
        except Exception as e: 
            os.chdir(pwd)
            raise e     
        print ("exists! loading...")
        return_object=folder_load(keys=keys)
    if verbose: print (filename," loaded from ",os.getcwd())
    os.chdir(pwd)
    return return_object
def Pickle(python_object,filename,path=None,save_path=".",verbose=True,overwrite=True,append=False,extension='.pickle'):
    '''save <python_object> to <filename> at location <save_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: save_path=path
    pwd=os.getcwd()
    if save_path != "." :
        os.chdir(save_path)
    if not overwrite:
        if filename in os.listdir("."): 
            raise IOError("File already exists!")
    if append: 
        assert type(python_object)==dict
        prev=Unpickle(filename)
        print_events(prev,name="old")
        python_object=merge_flat_dict(prev,python_object)
        print_events(python_object,name="appended")
    if type(python_object)==np.ndarray:
        np.save(filename,python_object)
        suffix=".npy"
    else:
        try:
            File=open(filename,"wb")
            pickle.dump(python_object,File)
        except OverflowError as e:
            File.close()
            os.system("rm "+filename)
            os.chdir(pwd)
            print (e,"trying to save as numpy arrays in folder...")
            folder_save(python_object,filename.split(".")[0],save_path)
            return
        suffix=""
    if verbose: print (filename+suffix, " saved at ", os.getcwd())
    os.chdir(pwd)
    return
def folder_save(events,folder_name,save_path,append=False):
    pwd=os.getcwd()
    os.chdir(save_path) 
    try: os.mkdir(folder_name)
    except FileExistsError as e: 
        print (e,"Overwriting...")
    finally:os.chdir(folder_name)                      
    for item in events: 
        if append:
            print ("appending...") 
            events[item]=np.concatenate((np.load(item+".npy",allow_pickle=True),events[item]),axis=0)
        if type(events[item]) ==list:
            print("list type found as val, creating directory...")
            os.mkdir(item)
            os.chdir(item)
            for i,array in enumerate(events[item]):
                np.save(item+str(i),array,allow_pickle=True)
                print (array.shape,"saved at ",os.getcwd())
            os.chdir("..")
        else: 
            np.save(item,events[item],allow_pickle=True)
            print (item+".npy saved at ",os.getcwd(), "shape = ",events[item].shape)
    os.chdir(pwd)
    return

def folder_load(keys=None,length=None):
    events=dict()
    pwd=os.getcwd()
    for filename in os.listdir("."):
        if os.path.isdir(filename):
            os.chdir(filename)
            events[filename]=[np.load(array_files,allow_pickle=True) for array_files in os.listdir(".")]
            os.chdir("..")  
            continue          
        if keys is not None:
            if filename[:-4] not in keys: continue
        try:
            events[filename[:-4]]=np.load(filename,allow_pickle=True)[:length]
        except IOError as e:
            os.chdir(pwd)
            raise e
        else:
            print (filename[:-4]," loaded to python dictionary...")
    return events
    
    

def transfer_weights(trained,model,verbose=False):
    enc_layers=model.layers
    count=0    
    for layer in trained.layers:
        temp=layer.get_weights()
        if verbose:
            print ("temp",temp)
        model.layers[count].set_weights(temp)
        if verbose:
            print ("set",model.layers[count].get_weights())
        count+=1
        if count==len(enc_layers):
            break
    return model
    
    
    
    



