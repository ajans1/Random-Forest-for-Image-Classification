'''
Goal: To use a random forest classifier (using sklearn) to classify image data of people
into different pain levels.

This Python program uses the BioVid Pain database. 

Our dataset consists of 99 videos that each consist of 138 frames. 

So, there are 138 times 99 or 13662 frames in total. 

Each video is labelled with one of 5 pain levels ranging from 0 or baseline
to 4 or severe pain. 
'''
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import copy

def convert_frames_to_videos(Xs, Ys, f_per_video=138):
    # Assuming the frames are not shuffled when flattened into a video
    # Also assuming number of frames is same for all videos
    # Also assuming that the label does not change within each video
    n, r = Xs.shape
    Ys = np.reshape(Ys, (-1, f_per_video))
    Xs = np.reshape(Xs, (-1, f_per_video, r))  
    return Xs, Ys[:, 0], Ys.shape[0]


def flatten_videos(Xs, Ys):
    #Reshapes the video back into frames from videos
    r = Xs.shape[-1]
    fpv = Xs.shape[-2]   
    Xs = np.reshape(Xs, (-1, r))    
    Ys = np.tile(Ys.reshape(-1, 1), (1, fpv)).reshape(-1)   
    return Xs, Ys
    

def stratified_sample_videos (
    Xs, Ys, split_frac=[3., 3., 3.], return_inds=False):
    # Since the test data looks very similar to the training data
    # We need to make sure that the training and test data is stratified correctly.
    # This function makes sure that frames across the same video are not placed in different datasets.
    split_frac = np.array(split_frac) / np.sum(split_frac)
    nd = split_frac.shape[0]
    n, fpv, r = Xs.shape
    classes = np.unique(Ys)
    c_inds = {}
    d_inds = {}
    for c in classes:
        c_inds[c] = (Ys==c).nonzero()[0]
        nc = c_inds[c].shape[0]
        split_sections = np.asarray(np.cumsum(split_frac) * nc, dtype=int)
        d_inds[c] = np.array_split(
                    np.random.permutation(c_inds[c]), split_sections[:-1])
    dset_X = {d:[] for d in range(nd)}
    dset_Y = {d:[] for d in range(nd)}
    for c in d_inds:
        for di, inds in enumerate(d_inds[c]):
            Xd, Yd = Xs[inds, :, :], Ys[inds]
            dset_X[di].append(Xd)
            dset_Y[di].append(Yd)
    
    dsets = [(np.concatenate(dset_X[d], 0), np.concatenate(dset_Y[d], 0))
            for d in range(nd)]
    
    if return_inds:
        inds = {d:np.concatenate([d_inds[c][d] for c in classes], 0)
            for d in range(nd)}
        return dsets, d_inds
    else:
        return dsets
    

def load_data():
    # Loads data set into a numpy array
    with open("C:/Users/ajani/Desktop/Research/RCNN - Sunwoo/Data", 
              'rb') as f1:
        X = np.asarray(pickle.load(f1))
    X = X.reshape(len(X),-1)
    with open("C:/Users/ajani/Desktop/Research/RCNN - Sunwoo/Labels",
              'rb') as f2:
        Y = np.asarray(pickle.load(f2))
    Y = Y.reshape(-1,1)
    return X,Y


def unoptimized_test_train_metrics(flag,rf, X_Train, Y_Train, X_Val, Y_Val, 
                            X_Test,Y_Test,Y_Train_bin, Y_Val_bin, Y_Test_bin):  
    
    # Uses the Randomforest Classifier to make predictions on binary class as 
    # well as multiclass test and train datasets.
    if flag == "five":
        print("Five Class Classification:")
        print()
        print("Unoptimized Test Train Metrics:")
        print()
        print("Training Results:")
        Y_Pred_Train = rf.predict(X_Train)
        get_results_multiple(Y_Train, Y_Pred_Train)
        print("\n\n\n")
            # predictions for test
        Y_Pred_Val  = rf.predict(X_Val)
        print("Validation Results:")
        get_results_multiple(Y_Val, Y_Pred_Val)
        print("\n\n\n")
        Y_Pred_Test = rf.predict(X_Test)
        print("Test Results:")
        get_results_multiple(Y_Test, Y_Pred_Test)
        '''
            # training metrics
        print("Training Metrics:")
        print(sklearn.metrics.classification_report(y_true= Y_Train, y_pred= Y_Pred_Train))    
        print()    
        #validation metrics
        print("Validation metrics:")
        print(sklearn.metrics.classification_report(y_true = Y_Val, y_pred = Y_Pred_Val))        
            # test data metrics        
        print()
        print("Test data metrics:")
        print(sklearn.metrics.classification_report(y_true= Y_Test, y_pred= Y_Pred_Test))
        '''
    if flag == "bin":
        print("\n\n\n")
        print("Binary Classification:")
        print()
        print("Unoptimized Test Train Metrics:")
        print()
        print("Training Results:")
        Y_Pred_Train = rf.predict(X_Train)
        get_results_binary(Y_Train, Y_Pred_Train)
        print("\n\n\n")
            # predictions for test
        Y_Pred_Val  = rf.predict(X_Val)
        print("Validation Results:")
        get_results_binary(Y_Val, Y_Pred_Val)
        print("\n\n\n")
        Y_Pred_Test = rf.predict(X_Test)
        print("Test Results:")
        get_results_binary(Y_Test, Y_Pred_Test)
        '''
            # training metrics
        print(sklearn.metrics.classification_report(y_true= Y_Train_bin, y_pred= Y_Pred_Train))    
        print()    
        #validation metrics
        print("Validation metrics:")
        print(sklearn.metrics.classification_report(y_true = Y_Val_bin, y_pred = Y_Pred_Val))        
            # test data metrics        
        print()
        print("Test data metrics:")
        print(sklearn.metrics.classification_report(y_true= Y_Test_bin, y_pred= Y_Pred_Test))
        '''
        
    
def get_results_binary(a,b):
    num_PA0 = 0;num_PA1 = 0;num_PA2 = 0;num_PA3 = 0;num_PA4 = 0
    cor_PA0 = 0;cor_PA1 = 0;cor_PA2 = 0;cor_PA3 = 0;cor_PA4 = 0
    for i in range(len(a)):
        if a[i] == 0:
            num_PA0 += 1
            if a[i] == b[i]:
                cor_PA0 += 1

        elif a[i] == 1:
            num_PA1 += 1
            if b[i] == 1:
                cor_PA1 += 1

        elif a[i] == 2:
            num_PA2 += 1
            if b[i] == 1:
                cor_PA2 += 1

        elif a[i] == 3:
            num_PA3 += 1
            if b[i] == 1:
                cor_PA3 += 1

        elif a[i] == 4 :
            num_PA4 += 1
            if b[i] == 1:
                cor_PA4 += 1
    print('###############################')
    print('Original     /',    'Estimated')
    print('num of BL/ No Pain: ', num_PA0, 'BL/ No Pain: ', 
                        cor_PA0, 'PAIN: ', num_PA0 - cor_PA0)
    print('num of PA1: ', num_PA1, 'PA1: ', cor_PA1, 
                          'BL/ No Pain: ', num_PA1 - cor_PA1 )
    print('num of PA2: ', num_PA2, 'PA2: ', cor_PA2,
                          'BL/ No Pain: ', num_PA2 - cor_PA2 )
    print('num of PA3: ', num_PA3, 'PA3: ', cor_PA3,
                          'BL/ No Pain: ', num_PA3 - cor_PA3 )
    print('num of PA4: ', num_PA4, 'PA4: ', cor_PA4, 
                          'BL/ No Pain: ', num_PA4 - cor_PA4 )
    print('###############################')
    print('Final results')
    print('acc of PA1: ', float((cor_PA1 + cor_PA0)/(num_PA1 + num_PA0)))
    print('acc of PA2: ', float((cor_PA2 + cor_PA0)/(num_PA2 + num_PA0)))
    print('acc of PA3: ', float((cor_PA3 + cor_PA0)/(num_PA3 + num_PA0)))
    print('acc of PA4: ', float((cor_PA4 + cor_PA0)/(num_PA4 + num_PA0)))
    print('avg acc: ', float((cor_PA1 + cor_PA2 + cor_PA3  + cor_PA4 + 
            4*cor_PA0)/(num_PA1 + num_PA2 + num_PA3 + num_PA4 + 4*num_PA0)))

    
def get_results_multiple(a,b):
    num_PA0 = 0;num_PA1 = 0;num_PA2 = 0;num_PA3 = 0;num_PA4 = 0
    cor_PA0 = 0;cor_PA1 = 0;cor_PA2 = 0;cor_PA3 = 0;cor_PA4 = 0
    for i in range(len(a)):
        if a[i] == 0:
            num_PA0 += 1
            if a[i] == b[i] and b[i] == 0:
                cor_PA0 += 1

        elif a[i] == 1:
            num_PA1 += 1
            if a[i] == b[i] and b[i] == 1:
                cor_PA1 += 1

        elif a[i] == 2:
            num_PA2 += 1
            if a[i] == b[i] and b[i] == 2:
                cor_PA2 += 1

        elif a[i] == 3:
            num_PA3 += 1
            if a[i] == b[i] and b[i] == 3:
                cor_PA3 += 1

        elif a[i] == 4:
            num_PA4 += 1
            if a[i] == b[i] and b[i] == 4:
                cor_PA4 += 1
    print('###############################')
    print('Original                 /',    '           Estimated')
    print('num of BL: ', num_PA0, 'BL: ', cor_PA0, 
          'Wrongly Classified: ', num_PA0 - cor_PA0)
    print('num of PA1: ', num_PA1, 'PA1: ', cor_PA1, 
          'Wrongly Classified: ', num_PA1 - cor_PA1 )
    print('num of PA2: ', num_PA2, 'PA2: ', cor_PA2, 
          'Wrongly Classified: ', num_PA2 - cor_PA2 )
    print('num of PA3: ', num_PA3, 'PA3: ', cor_PA3, 
          'Wrongly Classified: ', num_PA3 - cor_PA3 )
    print('num of PA4: ', num_PA4, 'PA4: ', cor_PA4, 
          'Wrongly Classified: ', num_PA4 - cor_PA4 )
    print('###############################')
    print('Final results')
    print('acc of BL:  ', float((cor_PA0)/(num_PA0)))
    print('acc of PA1: ', float((cor_PA1)/(num_PA1)))
    print('acc of PA2: ', float((cor_PA2)/(num_PA2)))
    print('acc of PA3: ', float((cor_PA3)/(num_PA3)))
    print('acc of PA4: ', float((cor_PA4)/(num_PA4)))
    print('avg acc: ', float((cor_PA1 + cor_PA2 + cor_PA3  + 
    cor_PA4 + cor_PA0)/(num_PA1 + num_PA2 + num_PA3 + num_PA4 + num_PA0)))



def cv_grid_search_metrics(rf,X_Train,Y_Train, X_Test, Y_Test):   

    clf = GridSearchCV(rf, param_grid={'n_estimators':[100],'min_samples_leaf':[3]})
    model = clf.fit(X_Train,Y_Train)
    
    
    y_pred_train = model.predict(X_Train)
        # predictions for test
    y_pred_test = model.predict(X_Test)
        # training metrics
    print("Training metrics:")
    print(sklearn.metrics.classification_report(y_true= Y_Train, y_pred= y_pred_train))
        
        # test data metrics
    print("Test data metrics:")
    print(sklearn.metrics.classification_report(y_true= Y_Test, y_pred= y_pred_test))
    
    return model



def ten_fold_cross_cv(model, X_Train, Y_Train_Original,Y_Train_Comparison,flag):

    sk_fold = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in sk_fold.split(X_Train,Y_Train_Original):
        train = X_Train[train_index]
        y_trn_k = Y_Train_Original[train_index]
        test = X_Train[test_index]
        y_tst_k = Y_Train_Original[test_index]
        # predictions for train
        #import IPython
        #IPython.embed()
        model.fit(train, y_trn_k)
        y_pred_train = model.predict(train)
        # predictions for test
        y_pred_test = model.predict(test)
        # training metrics
        if flag == "five": 
            print('###############################')
            print('    5 Class Classification:    ')
            print('###############################')
            print("Training metrics:") 
            get_results_multiple(y_trn_k, y_pred_train)
            # test data metrics
            print("Test data metrics:")
            get_results_multiple(y_tst_k, y_pred_test)
        if flag == "bin":
            print('###############################')
            print('Binary Classification Metrics: ')
            print('###############################')
            y_trn_k_comp = Y_Train_Comparison[train_index]
            y_tst_k_comp = Y_Train_Comparison[test_index]
            print("Training metrics:") 
            get_results_binary(y_trn_k_comp, y_pred_train)
            # test data metrics
            print("Test data metrics:")
            get_results_binary(y_tst_k_comp, y_pred_test)


def shuffle_datasets(dsets):
    
     
    X_Train, Y_Train = dsets[0][0],dsets[0][1]    
    X_Val, Y_Val = dsets[1][0],dsets[1][1]    
    X_Test, Y_Test = dsets[2][0],dsets[2][1]
    n = X_Train.shape[0]
    shuffle_indices = np.random.permutation(n)
    X_Train, Y_Train = X_Train[shuffle_indices],Y_Train[shuffle_indices]
    p = X_Val.shape[0]
    shuffle_indices2 = np.random.permutation(p)
    X_Val, Y_Val = X_Val[shuffle_indices2], Y_Val[shuffle_indices2]
    r = X_Test.shape[0]
    shuffle_indices3 = np.random.permutation(r)
    X_Test, Y_Test = X_Test[shuffle_indices3], Y_Test[shuffle_indices3]

    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test
    
    


def unoptimized_random_forest_classifier(Xs,Ys):
    fpv = 138
    Xs, Ys, _ = convert_frames_to_videos(Xs, Ys, fpv)
    fracs = [0.7, 0.15, 0.15]       
    dsets = stratified_sample_videos(Xs, Ys, fracs, False)
    dsets = [flatten_videos(*ds) for ds in dsets]
    X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = shuffle_datasets(dsets)
    rf1 = RandomForestClassifier()
    rf1.fit(X_Train, Y_Train)
    unoptimized_test_train_metrics(
            "five",rf1,X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test,[],[],[])
    rf2 = RandomForestClassifier()
    Y_Train_Bin = convert_to_binary(Y_Train)
    Y_Val_Bin   = convert_to_binary(Y_Val)
    Y_Test_Bin  = convert_to_binary(Y_Test)
    rf2.fit(X_Train,Y_Train_Bin)
    unoptimized_test_train_metrics(
    "bin",rf2,X_Train, Y_Train,X_Val, Y_Val, X_Test, Y_Test,Y_Train_Bin,Y_Val_Bin, Y_Test_Bin)
    return X_Train, Y_Train, X_Val, Y_Val,X_Test, Y_Test,Y_Train_Bin,Y_Val_Bin, Y_Test_Bin,rf1,rf2
    

def convert_to_binary(Y):
    Y_Bin = copy.copy(Y)
    for i in range (len(Y)):
        if Y[i] > 0:
            Y_Bin[i] = 1
        else: 
            Y_Bin[i] = 0
        
    return Y_Bin


if __name__ == "__main__":
    Xs, Ys = load_data() 
    X_Train, Y_Train, X_Val, Y_Val, X_Test,Y_Test,Y_Train_Bin,Y_Val_Bin, Y_Test_Bin, rf1, rf2= unoptimized_random_forest_classifier(Xs,Ys)
    model1 = cv_grid_search_metrics(rf1,X_Train,Y_Train, X_Test, Y_Test) 
    model2 = cv_grid_search_metrics(rf2,X_Train, Y_Train_Bin, X_Test, Y_Test_Bin)
    ten_fold_cross_cv(rf1, X_Train, Y_Train,[],"five")
    ten_fold_cross_cv(rf2,X_Train, Y_Train_Bin,Y_Train, "bin")
        
        
        
            