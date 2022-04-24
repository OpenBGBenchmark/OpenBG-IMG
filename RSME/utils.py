import numpy as np
import os
import random
import pickle
import math


def get_img_vec_array(proportion,img_vec_path='data/OpenBG/openbg_vit_best_img_vec.pickle',eutput_file='img_vec_id_fb15k_{}_vit.pickle',dim=1000):
    img_vec=pickle.load(open(img_vec_path,'rb'))
    img_vec={k.split('/')[-2]:v for k,v in img_vec.items()}
    # f=open('./src_data/FB15K_{}/ent_id'.format(proportion),'r')
    f=open('./data/OpenBG-IMG/ent_id', 'r')
    Lines=f.readlines()

    id2ent={}
    img_vec_array=[]
    for l in Lines:
        ent,id=l.strip().split()
        id2ent[id]=ent
        if ent.replace('/','.') in img_vec.keys():
            # print(id,ent)
            # id2vec[id]=img_vec[ent.replace('/','.')[1:]]
            img_vec_array.append(img_vec[ent.replace('/','.')])
        else:
            img_vec_array.append([0 for i in range(dim)])
    img_vec_by_id=np.array(img_vec_array)
    # out=open(eutput_file.format(proportion),'wb')
    out=open(eutput_file,'wb')
    pickle.dump(img_vec_by_id,out)
    out.close()


def get_img_vec_array_forget(proportion,remember_proportion,rank_file='openbg_vit_rank.txt',eutput_file='rel_MPR_PD_vit_{}_mrp{}.pickle'):
    with open(rank_file,'r') as f:
        Ranks=f.readlines()
        rel_rank={}
        for r in Ranks[:-1]:
            try:
                rel,_,mrp=r.strip().split('\t')
            except Exception as e:
                # # print(e)
                # # print(r)
                continue
            rel_rank[rel[13:]]=float(mrp[12:])

    # with open('./data/FB15K_{}/rel_id'.format(proportion),'r') as f:
    with open('./data/OpenBG-IMG/rel_id', 'r') as f:
        Lines=f.readlines()

    rel_id_pd=[]
    for l in Lines:
        rel,_=l.strip().split('\t')
        try:
            if rel_rank[rel]<remember_proportion/100.0:
                rel_id_pd.append([1])
            else:
                rel_id_pd.append([0])
        except Exception as e:
            print(e)
            rel_id_pd.append([0])
            continue

    rel_id_pd=np.array(rel_id_pd)

    with open(eutput_file.format(remember_proportion),'wb') as out:
        pickle.dump(rel_id_pd,out)


def get_img_vec_sig_alpha(proportion,rank_file='openbg_vit_rank.txt',eutput_file='rel_MPR_SIG_vit_{}.pickle'):
    with open(rank_file,'r') as f:
        Ranks=f.readlines()[:-1]
        rel_rank={}
        for r in Ranks:
            try:
                rel,_,mrp=r.strip().split('\t')
            except Exception as e:
                print(e)
                print(r)
                continue
            rel_rank[rel[13:]]=float(mrp[12:])

    # with open('./data/FB15K_{}/rel_id'.format(proportion),'r') as f:
    with open('./data/OpenBG-IMG/rel_id', 'r') as f:
        Lines=f.readlines()

    rel_sig_alpha=[]
    for l in Lines:
        rel,_=l.strip().split('\t')
        try:
            rel_sig_alpha.append([1/(1+math.exp(rel_rank[rel]))])
        except Exception as e:
            print(e)
            rel_sig_alpha.append([1 / (1 + math.exp(1))])
            continue

    rel_id_pd=np.array(rel_sig_alpha)

    with open(eutput_file,'wb') as out:
        pickle.dump(rel_id_pd,out)

def sample(proportion,data_path='./src_data/OpenBG'):
    with open(data_path+'/train') as f:
        Ls=f.readlines()
        L = [random.randint(0, len(Ls)-1) for _ in range(round(len(Ls)*proportion))]
        Lf=[Ls[l] for l in L]

    if not os.path.exists(data_path+'_{}/'.format(round(proportion*100))):
        os.mkdir(data_path+'_{}/'.format(round(proportion*100)))
    Ent=set()

    with open(data_path+'_{}/train'.format(round(100*proportion)),'w') as f:
        for l in Lf:
            h,r,t=l.strip().split()
            Ent.add(h)
            Ent.add(r)
            Ent.add(t)
            f.write(l)
            f.flush()

    with open(data_path+'/valid','r') as f:
        Ls = f.readlines()

    with open(data_path+'_{}/valid'.format(round(100*proportion)),'w') as f:
        for l in Ls:
            h,r,t=l.strip().split()
            if h in Ent and r in Ent and t in Ent:
                f.write(l)
                f.flush()
            else:
                print(l.strip()+' pass')

    with open(data_path+'/test','r') as f:
        Ls = f.readlines()

    with open(data_path+'_{}/test'.format(round(proportion*100)),'w') as f:
        for l in Ls:
            h, r, t = l.strip().split()
            if h in Ent and r in Ent and t in Ent:
                f.write(l)
                f.flush()
            else:
                print(l.strip()+' pass')

if __name__ == '__main__':
    # sample(0.2)
    get_img_vec_array(0, img_vec_path='./data/OpenBG-IMG/openbg_vit_best_img_vec.pickle', eutput_file='./data/OpenBG-IMG/img_vec_id_openbg_vit.pickle')
    get_img_vec_sig_alpha(20, './data/OpenBG-IMG/openbg_vit_rank.txt', './data/OpenBG-IMG/rel_MPR_SIG_vit.pickle')
    get_img_vec_array_forget(30, 100, './data/OpenBG-IMG/openbg_vit_rank.txt', './data/OpenBG-IMG/rel_MPR_PD_vit_mrp{}.pickle')
    pass



