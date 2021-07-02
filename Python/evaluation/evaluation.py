from ..cgmlvq import CGMLVQ
from math import sqrt
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import KFold

import csv
import numpy as np
import os
import unittest


# wrap in test class to start with intellij
class Evaluate_CGMLVQ( unittest.TestCase ):

    KNN_2CLASS_DATASET = "voting_sim.csv"
    CGMLVQ_2CLASS_DATASET = "voting_sim_embedded.csv"

    KNN_MULTICLASS_DATASET = "zongker_sim.csv"
    CGMLVQ_MULTICLASS_DATASET = "zongker_sim_embedded.csv"

    # 2 class dataset
    # kf = KFold( n_splits=5, random_state=None, shuffle=False )

    # multiclass dataset
    kf = KFold( n_splits=5, random_state=1, shuffle=True )


    def test_evaluate( self ):

        # self.knn( self.KNN_2CLASS_DATASET, True )
        # self.cgmlvq( self.CGMLVQ_2CLASS_DATASET, True )

        self.knn( self.KNN_MULTICLASS_DATASET, False )
        self.cgmlvq( self.CGMLVQ_MULTICLASS_DATASET, False )


    def knn_euclidean_distance( self, row1, row2 ):

        distance = 0.0

        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2

        return sqrt(distance)


    def knn_get_neighbors( self, train, test_row, num_neighbors ):

        # Locate the most similar neighbors

        distances = list()

        for train_row in train:
            dist = self.knn_euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))

        distances.sort(key=lambda tup: tup[1])

        neighbors = list()

        for i in range(num_neighbors):
            neighbors.append(distances[i][0])

        return neighbors


    def knn_predict_classification( self, train, test_row, num_neighbors ):

        # Make a classification prediction with neighbors

        neighbors = self.knn_get_neighbors(train, test_row, num_neighbors)

        output_values = [row[-1] for row in neighbors]

        prediction = max(set(output_values), key=output_values.count)

        return prediction


    def knn_predict( self, train, test, neighbors ):

        predicted = []

        for test_row in test:
            predicted.append( self.knn_predict_classification(train, test_row, neighbors) )

        return predicted


    def load_dataset( self, dataset ):

        data = []

        csv_file = open( os.path.join(os.getcwd(), "Python\\data sets", dataset) )

        csv_reader = csv.reader( csv_file, delimiter=',' )

        for row in csv_reader:
            data.append( row )

        csv_file.close()

        return data


    def cgmlvq( self, dataset, isBinary ):

        data = np.array( self.load_dataset(dataset) )

        X = np.array( data[:,:-1], dtype=np.cdouble )
        y = np.array( data[:,-1], dtype=int )

        for train_index, test_index in self.kf.split( X ):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print( f" Aufteilung Durchgang: {np.asarray(np.unique(y_train,return_counts=True)).T}" )

            self.cgmlvq_run( X_train, X_test, y_train, y_test, isBinary )


    def knn( self, dataset, isBinary ):

        data = np.array( self.load_dataset(dataset) )

        for train_index, test_index in self.kf.split( data ):

            train = np.array(data[train_index], dtype=float)
            test = np.array(data[test_index], dtype=float)
            y_test = np.array(data[test_index,-1], dtype=int)

            print( f" Aufteilung Durchgang: {np.asarray(np.unique(train[:,-1],return_counts=True)).T}" )

            self.knn_run( train, test, y_test, isBinary )


    def knn_run( self, train, test, y_test, isBinary ):

        # kNN source
        # https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

        predicted = self.knn_predict( train, test, int(sqrt(len(train))) )

        print( f"kNN (k={int(sqrt(len(train)))})" )
        print( f"---" )

        if isBinary:
            self.binary_metrics( y_test, predicted )
        else:
            self.multilabel_metrics( y_test, predicted )


    def cgmlvq_run( self, X_train, X_test, y_train, y_test, isBinary ):

        cgmlvq = CGMLVQ()
        cgmlvq.set_params()
        cgmlvq.fit( X_train, y_train )

        predicted = cgmlvq.predict( X_test )

        print( f"CGMLVQ ({cgmlvq.get_params()})" )
        print( f"------" )

        if isBinary:
            self.binary_metrics( y_test, predicted )
        else:
            self.multilabel_metrics( y_test, predicted )


    def binary_metrics( self, y_test, predicted ):

        fpr, tpr, thresholds = roc_curve( y_test-1, predicted-1 )  # must be 0 and 1 labels

        print( f"cm: {confusion_matrix(y_test, predicted)}" )
        print( f"acc: {accuracy_score(y_test, predicted)}" )
        print( f"prec: {precision_score(y_test, predicted)}" )
        print( f"rec: {recall_score(y_test, predicted)}" )
        print( f"f1: {f1_score(y_test, predicted)}" )
        print( f"auc: {auc(fpr, tpr)}" )


    def multilabel_metrics( self, y_test, predicted ):

        print( f"cm: {confusion_matrix(y_test, predicted)}" )

        print( f"acc: {accuracy_score(y_test, predicted)}" )

        print( f"prec mi: {precision_score(y_test, predicted, average='micro')}" )
        print( f"prec ma: {precision_score(y_test, predicted, average='macro')}" )
        print( f"prec we: {precision_score(y_test, predicted, average='weighted')}" )

        print( f"rec mi: {recall_score(y_test, predicted, average='micro')}" )
        print( f"rec ma: {recall_score(y_test, predicted, average='macro')}" )
        print( f"rec we: {recall_score(y_test, predicted, average='weighted')}" )

        print( f"f1 mi: {f1_score(y_test, predicted, average='micro')}" )
        print( f"f1 ma: {f1_score(y_test, predicted, average='macro')}" )
        print( f"f1 we: {f1_score(y_test, predicted, average='weighted')}" )


    # =========================
    # =========================
    # Data set: voting_sim
    # Datensätze: 435
    # Feature vectors: 435
    # Verteilung: Klasse 1: 267
    #             Klasse 2: 168
    # Kreuzvalidierung: 5
    # =========================
    # =========================
    #
    #
    # Durchgang 1: Klasse 1: 215
    #              Klasse 2: 133
    #
    # kNN (k=1)
    # ---
    # cm: [[50  2]
    #      [ 0 35]]
    # acc: 0.9770114942528736
    # prec: 1.0
    # rec: 0.9615384615384616
    # f1: 0.9803921568627451
    # auc: 0.9807692307692308
    #
    # kNN (k=18)
    # ---
    # cm: [[49  3]
    #      [ 1 34]]
    # acc: 0.9540229885057471
    # prec: 0.98
    # rec: 0.9423076923076923
    # f1: 0.9607843137254902
    # auc: 0.9568681318681319
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[50  2]
    #      [ 2 33]]
    # acc: 0.9540229885057471
    # prec: 0.9615384615384616
    # rec: 0.9615384615384616
    # f1: 0.9615384615384616
    # auc: 0.9521978021978023
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[50  2]
    #      [ 2 33]]
    # acc: 0.9540229885057471
    # prec: 0.9615384615384616
    # rec: 0.9615384615384616
    # f1: 0.9615384615384616
    # auc: 0.9521978021978023
    #
    # CGMLVQ ({'coefficients': 435, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[53  3]
    #      [ 1 30]]
    # acc: 0.9540229885057471
    # prec: 0.9814814814814815
    # rec: 0.9464285714285714
    # f1: 0.9636363636363636
    # auc: 0.9570852534562212
    #
    # CGMLVQ ({'coefficients': 435, 'totalsteps': 100, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[53  3]
    #      [ 1 30]]
    # acc: 0.9540229885057471
    # prec: 0.9814814814814815
    # rec: 0.9464285714285714
    # f1: 0.9636363636363636
    # auc: 0.9570852534562212
    #
    #
    # Durchgang 2: Klasse 1: 214
    #              Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[48  5]
    #      [ 1 33]]
    # acc: 0.9310344827586207
    # prec: 0.9795918367346939
    # rec: 0.9056603773584906
    # f1: 0.9411764705882353
    # auc: 0.9381243063263042
    #
    # kNN (k=18)
    # ---
    # cm: [[50  3]
    #      [ 1 33]]
    # acc: 0.9540229885057471
    # prec: 0.9803921568627451
    # rec: 0.9433962264150944
    # f1: 0.9615384615384616
    # auc: 0.956992230854606
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[46  7]
    #      [ 0 34]]
    # acc: 0.9195402298850575
    # prec: 1.0
    # rec: 0.8679245283018868
    # f1: 0.9292929292929293
    # auc: 0.9339622641509434
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[46  7]
    #      [ 1 33]]
    # acc: 0.9080459770114943
    # prec: 0.9787234042553191
    # rec: 0.8679245283018868
    # f1: 0.9199999999999999
    # auc: 0.9192563817980023
    #
    #
    # Aufteilung Durchgang 3: Klasse 1: 211
    #                         Klasse 2: 137
    #
    # kNN (k=1)
    # ---
    # cm: [[54  2]
    #      [ 3 28]]
    # acc: 0.9425287356321839
    # prec: 0.9473684210526315
    # rec: 0.9642857142857143
    # f1: 0.9557522123893805
    # auc: 0.9337557603686636
    #
    # kNN (k=18)
    # ---
    # cm: [[55  1]
    #      [ 4 27]]
    # acc: 0.9425287356321839
    # prec: 0.9322033898305084
    # rec: 0.9821428571428571
    # f1: 0.9565217391304348
    # auc: 0.9265552995391705
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[55  1]
    #      [ 2 29]]
    # acc: 0.9655172413793104
    # prec: 0.9649122807017544
    # rec: 0.9821428571428571
    # f1: 0.9734513274336283
    # auc: 0.9588133640552995
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[55  1]
    #      [ 3 28]]
    # acc: 0.9540229885057471
    # prec: 0.9482758620689655
    # rec: 0.9821428571428571
    # f1: 0.9649122807017544
    # auc: 0.942684331797235
    #
    #
    # Aufteilung Durchgang 4: Klasse 1: 214
    #                         Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[52  1]
    #      [ 5 29]]
    # acc: 0.9310344827586207
    # prec: 0.9122807017543859
    # rec: 0.9811320754716981
    # f1: 0.9454545454545454
    # auc: 0.9170366259711432
    #
    # kNN (k=18)
    # ---
    # cm: [[53  0]
    #      [ 1 33]]
    # acc: 0.9885057471264368
    # prec: 0.9814814814814815
    # rec: 1.0
    # f1: 0.9906542056074767
    # auc: 0.9852941176470589
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[51  2]
    #      [ 3 31]]
    # acc: 0.9425287356321839
    # prec: 0.9444444444444444
    # rec: 0.9622641509433962
    # f1: 0.9532710280373832
    # auc: 0.9370144284128745
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[52  1]
    #      [ 3 31]]
    # acc: 0.9540229885057471
    # prec: 0.9454545454545454
    # rec: 0.9811320754716981
    # f1: 0.9629629629629629
    # auc: 0.9464483906770255
    #
    #
    # Aufteilung Durchgang 5: Klasse 1: 214
    #                         Klasse 2: 134
    #
    # kNN (k=1)
    # ---
    # cm: [[46  7]
    #      [ 3 31]]
    # acc: 0.8850574712643678
    # prec: 0.9387755102040817
    # rec: 0.8679245283018868
    # f1: 0.9019607843137256
    # auc: 0.8898446170921198
    #
    # kNN (k=18)
    # ---
    # cm: [[47  6]
    #      [ 4 30]]
    # acc: 0.8850574712643678
    # prec: 0.9215686274509803
    # rec: 0.8867924528301887
    # f1: 0.9038461538461539
    # auc: 0.8845726970033296
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[45  8]
    #      [ 3 31]]
    # acc: 0.8735632183908046
    # prec: 0.9375
    # rec: 0.8490566037735849
    # f1: 0.8910891089108911
    # auc: 0.8804106548279689
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[45  8]
    #      [ 4 30]]
    # acc: 0.8620689655172413
    # prec: 0.9183673469387755
    # rec: 0.8490566037735849
    # f1: 0.8823529411764707
    # auc: 0.8657047724750278


    # =========================
    # =========================
    # Data set: zongker_sim
    # Datensätze: 2000
    # Feature vectors: 2000
    # Klassenverteilung: 10 Klassen, je 200
    # Kreuzvalidierung: 5
    # =========================
    # =========================
    #
    #
    # Durchgang 1: Klasse 1: 161
    #              Klasse 2: 158
    #              Klasse 3: 164
    #              Klasse 4: 161
    #              Klasse 5: 161
    #              Klasse 6: 160
    #              Klasse 7: 159
    #              Klasse 8: 161
    #              Klasse 9: 155
    #              Klasse 10: 160
    #
    # kNN (k=1)
    # ---
    # cm: [[38  0  0  0  0  0  0  0  1  0]
    #      [ 0 41  0  0  0  0  0  0  1  0]
    #      [ 0  0 31  3  0  0  1  0  1  0]
    #      [ 0  0  0 37  0  1  1  0  0  0]
    #      [ 0  0  0  0 38  0  0  0  0  1]
    #      [ 0  0  0  1  0 39  0  0  0  0]
    #      [ 1  1  0  0  0  2 37  0  0  0]
    #      [ 0  0  0  0  0  0  0 39  0  0]
    #      [ 2  0  0  0  0  0  0  0 43  0]
    #      [ 1  0  0  0  0  0  0  1  0 38]]
    # acc: 0.9525
    # prec mi: 0.9525
    # prec ma: 0.9544822365686629
    # prec we: 0.9539642641072252
    # rec mi: 0.9525
    # rec ma: 0.9517732064683283
    # rec we: 0.9525
    # f1 mi: 0.9525
    # f1 ma: 0.9522489748327754
    # f1 we: 0.952407816773373
    #
    # kNN (k=40)
    # ---
    # cm: [[38  0  0  0  0  0  0  0  1  0]
    #      [ 0 41  0  0  0  0  0  0  1  0]
    #      [ 0  0 29  4  0  0  0  0  3  0]
    #      [ 1  0  0 36  0  0  1  0  1  0]
    #      [ 1  0  0  0 33  0  0  0  1  4]
    #      [ 1  0  0  3  0 34  0  0  2  0]
    #      [ 1  1  3  1  0  0 34  0  1  0]
    #      [ 0  0  0  0  0  0  0 36  1  2]
    #      [ 0  0  1  1  0  0  1  0 42  0]
    #      [ 2  0  0  0  0  0  0  1  2 35]]
    # acc: 0.895
    # prec mi: 0.895
    # prec ma: 0.9053327036253865
    # prec we: 0.9042408195335025
    # rec mi: 0.895
    # rec ma: 0.8936014324428958
    # rec we: 0.895
    # f1 mi: 0.895
    # f1 ma: 0.8959844114699967
    # f1 we: 0.8960752574686309
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[ 7  8  6  1  4  7  0  0  0  6]
    #      [ 7  7  7  1  4  4  0 10  1  1]
    #      [ 4  6  9  2  2  4  0  2  0  7]
    #      [ 3  3  5  3  3  6  3  0  3 10]
    #      [ 7  7  8  1  3  9  0  0  1  3]
    #      [ 6  5  7  0  6 11  0  0  0  5]
    #      [ 6  5  4  0  3 14  1  4  1  3]
    #      [ 3  9  3  0  2  3  0 13  2  4]
    #      [ 2  6 11  5  2  3  0  5  7  4]
    #      [ 7  5  5  1  4  5  0  4  0  9]]
    # acc: 0.175
    # prec mi: 0.175
    # prec ma: 0.20915413462005347
    # prec we: 0.21284683442815452
    # rec mi: 0.175
    # rec ma: 0.1763279132791328
    # rec we: 0.175
    # f1 mi: 0.175
    # f1 ma: 0.16831666453930438
    # f1 we: 0.16852175231971672
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': False, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[ 5  7  8  4  2  5  0  0  1  7]
    #      [ 8  9  8  2  2  3  0  8  1  1]
    #      [ 1  6 14  2  2  6  0  0  2  3]
    #      [ 2  5  6  2  4  9  0  0  1 10]
    #      [ 6  8  8  2  3  7  0  0  2  3]
    #      [ 5  5  7  1  5 11  0  0  0  6]
    #      [ 5  8  6  1  5  7  0  2  3  4]
    #      [ 3  9  5  2  0  2  0 11  2  5]
    #      [ 1  7 12  4  2  1  0  3  7  8]
    #      [ 6  7  5  2  2  2  0  3  4  9]]
    # acc: 0.1775
    # prec mi: 0.1775
    # prec ma: 0.1705060263341491
    # prec we: 0.1713508370072152
    # rec mi: 0.1775
    # rec ma: 0.1797191697191697
    # rec we: 0.1775
    # f1 mi: 0.17750000000000002
    # f1 ma: 0.16459847738516026
    # f1 we: 0.16400050190307625
    #
    # CGMLVQ ({'coefficients': 100, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[33  0  2  0  0  2  0  0  2  0]
    #      [ 0 39  0  0  1  0  1  0  1  0]
    #      [ 6  1 20  3  0  1  2  1  2  0]
    #      [ 3  0  4 25  0  2  1  1  2  1]
    #      [ 2  0  0  0 24  0  7  2  0  4]
    #      [ 6  0  0  7  0 22  2  0  3  0]
    #      [ 3  4  0  0  2  2 28  0  1  1]
    #      [ 0  2  0  2  0  0  0 28  2  5]
    #      [ 3  1  0  3  0  1  0  0 33  4]
    #      [ 2  0  0  1  5  0  1  4  4 23]]
    # acc: 0.6875
    # prec mi: 0.6875
    # prec ma: 0.6970780553748193
    # prec we: 0.6966851020379405
    # rec mi: 0.6875
    # rec ma: 0.6845899967241431
    # rec we: 0.6875
    # f1 mi: 0.6875
    # f1 ma: 0.6837451816134463
    # f1 we: 0.6852262102568261
    #
    # CGMLVQ ({'coefficients': 100, 'totalsteps': 100, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[35  0  1  0  0  1  1  0  0  1]
    #      [ 1 38  0  0  1  0  1  0  1  0]
    #      [ 6  0 22  1  0  3  2  1  1  0]
    #      [ 2  0  4 27  0  4  1  0  1  0]
    #      [ 1  0  1  0 27  0  5  1  0  4]
    #      [ 4  0  1  4  0 26  2  0  3  0]
    #      [ 2  4  0  1  0  2 31  0  0  1]
    #      [ 0  1  0  1  0  0  0 30  2  5]
    #      [ 3  0  0  2  0  1  0  1 33  5]
    #      [ 2  0  0  1  1  0  0  3  3 30]]
    # acc: 0.7475
    # prec mi: 0.7475
    # prec ma: 0.7587246014013735
    # prec we: 0.7589365808728258
    # rec mi: 0.7475
    # rec ma: 0.7456585961464011
    # rec we: 0.7475
    # f1 mi: 0.7475
    # f1 ma: 0.7465194157825785
    # f1 we: 0.7477319590949497
    #
    # CGMLVQ ({'coefficients': 100, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[36  0  1  0  0  0  1  0  0  1]
    #      [ 1 37  0  0  1  0  1  0  1  1]
    #      [ 3  0 25  1  0  4  1  1  1  0]
    #      [ 1  0  3 26  0  5  1  0  2  1]
    #      [ 1  0  1  0 27  0  4  1  1  4]
    #      [ 4  0  1  1  0 28  1  0  4  1]
    #      [ 1  4  0  0  1  2 32  0  0  1]
    #      [ 0  0  0  1  0  0  0 31  1  6]
    #      [ 4  0  0  2  0  2  1  1 33  2]
    #      [ 1  0  0  0  2  0  0  3  3 31]]
    # acc: 0.765
    # prec mi: 0.765
    # prec ma: 0.7756769815648051
    # prec we: 0.774897256393083
    # rec mi: 0.765
    # rec ma: 0.7641141040531285
    # rec we: 0.765
    # f1 mi: 0.765
    # f1 ma: 0.7651381442816311
    # f1 we: 0.7653237240011289
    #
    # CGMLVQ ({'coefficients': 200, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[37  0  0  0  0  1  1  0  0  0]
    #      [ 0 41  0  0  0  0  1  0  0  0]
    #      [ 2  0 30  1  0  0  1  1  1  0]
    #      [ 2  0  2 31  0  1  1  0  2  0]
    #      [ 2  0  1  0 29  0  0  0  1  6]
    #      [ 2  0  0  3  0 29  2  0  3  1]
    #      [ 3  1  0  1  1  2 33  0  0  0]
    #      [ 0  0  0  0  0  0  0 37  0  2]
    #      [ 2  0  1  0  0  1  1  0 39  1]
    #      [ 2  0  0  0  1  0  0  2  1 34]]
    # acc: 0.85
    # prec mi: 0.85
    # prec ma: 0.8572132544224675
    # prec we: 0.8571226842081439
    # rec mi: 0.85
    # rec ma: 0.8491965960868401
    # rec we: 0.85
    # f1 mi: 0.85
    # f1 ma: 0.8494415597596665
    # f1 we: 0.849873112270223
    #
    # CGMLVQ ({'coefficients': 400, 'totalsteps': 400, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[37  0  0  0  0  0  1  0  1  0]
    #      [ 0 40  0  0  0  0  0  0  2  0]
    #      [ 1  0 30  2  0  0  1  1  1  0]
    #      [ 2  0  2 31  0  1  1  1  1  0]
    #      [ 0  0  0  0 36  0  0  0  1  2]
    #      [ 1  0  0  3  0 34  2  0  0  0]
    #      [ 1  1  0  0  0  2 34  0  3  0]
    #      [ 0  0  0  0  0  0  0 37  0  2]
    #      [ 1  0  0  2  0  0  2  0 40  0]
    #      [ 1  0  0  0  0  0  0  1  4 34]]
    # acc: 0.8825
    # prec mi: 0.8825
    # prec ma: 0.8892449355530048
    # prec we: 0.8873008709178675
    # rec mi: 0.8825
    # rec ma: 0.8819256082670716
    # rec we: 0.8825
    # f1 mi: 0.8825
    # f1 ma: 0.8840184872084684
    # f1 we: 0.8833078124419614
    #
    # CGMLVQ ({'coefficients': 1000, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[39  0  0  0  0  0  0  0  0  0]
    #      [ 0 42  0  0  0  0  0  0  0  0]
    #      [ 1  0 31  2  0  1  1  0  0  0]
    #      [ 1  0  1 34  0  1  1  0  1  0]
    #      [ 0  0  0  0 35  0  0  0  1  3]
    #      [ 0  0  0  1  0 38  0  0  1  0]
    #      [ 2  1  0  0  0  3 35  0  0  0]
    #      [ 0  0  0  0  0  0  0 38  0  1]
    #      [ 1  0  1  0  0  0  2  0 41  0]
    #      [ 1  0  0  0  0  0  0  1  1 37]]
    # acc: 0.925
    # prec mi: 0.925
    # prec ma: 0.9270789648554821
    # prec we: 0.9268013636243927
    # rec mi: 0.925
    # rec ma: 0.9244470502397333
    # rec we: 0.925
    # f1 mi: 0.925
    # f1 ma: 0.9245753218364042
    # f1 we: 0.9247483469720504
    #
    # CGMLVQ ({'coefficients': 2000, 'totalsteps': 100, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[39  0  0  0  0  0  0  0  0  0]
    #      [ 0 42  0  0  0  0  0  0  0  0]
    #      [ 2  0 30  3  0  0  1  0  0  0]
    #      [ 2  0  1 33  0  1  1  0  1  0]
    #      [ 0  0  0  0 32  0  1  0  1  5]
    #      [ 0  0  0  0  0 39  0  0  1  0]
    #      [ 2  1  0  0  0  3 34  0  1  0]
    #      [ 0  0  0  0  0  0  0 38  0  1]
    #      [ 1  0  0  1  0  1  1  0 41  0]
    #      [ 1  0  0  0  0  0  0  1  1 37]]
    # acc: 0.9125
    # prec mi: 0.9125
    # prec ma: 0.9173394164397859
    # prec we: 0.9166837692175354
    # rec mi: 0.9125
    # rec ma: 0.9114738378153013
    # rec we: 0.9125
    # f1 mi: 0.9125
    # f1 ma: 0.9116918992459631
    # f1 we: 0.9119655737791658
    #
    #
    # Durchgang 2: Klasse 1: 160
    #              Klasse 2: 164
    #              Klasse 3: 153
    #              Klasse 4: 161
    #              Klasse 5: 154
    #              Klasse 6: 162
    #              Klasse 7: 162
    #              Klasse 8: 161
    #              Klasse 9: 164
    #              Klasse 10: 159
    #
    # kNN (k=1)
    # ---
    # cm: [[39  0  0  0  0  0  1  0  0  0]
    #      [ 0 36  0  0  0  0  0  0  0  0]
    #      [ 0  0 43  1  0  1  0  0  2  0]
    #      [ 0  0  0 34  0  0  0  0  5  0]
    #      [ 0  0  0  1 43  0  0  0  0  2]
    #      [ 0  0  0  1  0 35  0  0  2  0]
    #      [ 0  0  0  2  0  0 36  0  0  0]
    #      [ 0  0  0  0  0  0  0 36  0  3]
    #      [ 0  0  0  1  0  0  0  0 34  1]
    #      [ 0  0  0  0  0  0  0  0  2 39]]
    # acc: 0.9375
    # prec mi: 0.9375
    # prec ma: 0.9417417417417419
    # prec we: 0.944501876876877
    # rec mi: 0.9375
    # rec ma: 0.9383633029859869
    # rec we: 0.9375
    # f1 mi: 0.9375
    # f1 ma: 0.9382377819181299
    # f1 we: 0.9392451378309872
    #
    # kNN (k=40)
    # ---
    # cm: [[37  0  0  0  0  0  1  0  2  0]
    #      [ 0 35  0  0  0  0  1  0  0  0]
    #      [ 1  0 41  0  0  1  1  0  3  0]
    #      [ 0  0  1 32  0  0  0  0  6  0]
    #      [ 0  0  1  1 37  0  0  0  1  6]
    #      [ 0  0  0  0  0 34  0  0  4  0]
    #      [ 2  0  0  0  0  0 36  0  0  0]
    #      [ 0  0  0  0  0  0  0 34  0  5]
    #      [ 0  0  0  0  0  0  0  0 35  1]
    #      [ 0  0  0  0  0  0  0  0  2 39]]
    # acc: 0.9
    # prec mi: 0.9
    # prec ma: 0.9167774077138994
    # prec we: 0.9193746754497342
    # rec mi: 0.9
    # rec ma: 0.9031765163724025
    # rec we: 0.9
    # f1 mi: 0.9
    # f1 ma: 0.9034903332602167
    # f1 we: 0.9033196651616427
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[ 4  6  9  7  9  2  0  2  0  1]
    #      [ 2 11  2  4  7  0  4  4  1  1]
    #      [ 7  6 10  6  5  0  3  2  1  7]
    #      [ 3  7  7  4  7  4  5  0  0  2]
    #      [ 5  5 14  5  6  2  2  3  1  3]
    #      [ 3  0  6  4 10  5  6  0  0  4]
    #      [ 2  5  6  5  6  6  3  0  0  5]
    #      [ 4 10  4  5  3  0  0  6  3  4]
    #      [ 2  8  9  1  4  2  2  3  1  4]
    #      [ 3  4  8  4  6  2  3  2  0  9]]
    # acc: 0.1475
    # prec mi: 0.1475
    # prec ma: 0.15742839636598402
    # prec we: 0.15602332493943644
    # rec mi: 0.1475
    # rec ma: 0.1462982840710519
    # rec we: 0.1475
    # f1 mi: 0.1475
    # f1 ma: 0.14207194059151113
    # f1 we: 0.14243359441214187
    #
    # CGMLVQ ({'coefficients': 1000, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[39  0  0  0  0  0  1  0  0  0]
    #      [ 0 36  0  0  0  0  0  0  0  0]
    #      [ 0  0 41  1  0  1  3  0  1  0]
    #      [ 0  0  2 31  0  2  0  0  4  0]
    #      [ 0  0  0  0 38  0  2  0  1  5]
    #      [ 1  0  0  1  0 36  0  0  0  0]
    #      [ 0  1  1  0  0  1 35  0  0  0]
    #      [ 0  0  0  0  0  0  0 36  0  3]
    #      [ 0  0  0  0  0  0  0  0 35  1]
    #      [ 0  0  0  0  0  0  0  0  0 41]]
    # acc: 0.92
    # prec mi: 0.92
    # prec ma: 0.9246502167355827
    # prec we: 0.9261239422904057
    # rec mi: 0.92
    # rec ma: 0.9232019374856172
    # rec we: 0.92
    # f1 mi: 0.92
    # f1 ma: 0.9207615969468753
    # f1 we: 0.9198025386107449
    #
    #
    # Durchgang 3: Klasse 1: 161
    #              Klasse 2: 159
    #              Klasse 3: 155
    #              Klasse 4: 159
    #              Klasse 5: 167
    #              Klasse 6: 158
    #              Klasse 7: 156
    #              Klasse 8: 161
    #              Klasse 9: 155
    #              Klasse 10: 169
    #
    # kNN (k=1)
    # ---
    # cm: [[38  0  0  0  0  0  0  0  0  1]
    #      [ 0 40  0  0  0  0  0  0  1  0]
    #      [ 0  0 41  2  0  0  0  0  2  0]
    #      [ 1  0  0 40  0  0  0  0  0  0]
    #      [ 0  0  0  0 33  0  0  0  0  0]
    #      [ 1  0  0  4  1 36  0  0  0  0]
    #      [ 0  0  0  0  0  0 44  0  0  0]
    #      [ 0  0  0  0  0  0  0 39  0  0]
    #      [ 2  0  0  1  0  0  0  0 41  1]
    #      [ 0  0  0  0  2  0  1  0  0 28]]
    # acc: 0.95
    # prec mi: 0.95
    # prec ma: 0.9515421694145099
    # prec we: 0.9532917626109116
    # rec mi: 0.95
    # rec ma: 0.9508169372370787
    # rec we: 0.95
    # f1 mi: 0.9500000000000001
    # f1 ma: 0.9496249015753022
    # f1 we: 0.9500647011470224
    #
    # kNN (k=40)
    # ---
    # cm: [[37  0  0  0  0  0  0  0  2  0]
    #      [ 0 40  0  0  0  0  0  0  1  0]
    #      [ 0  0 38  0  0  0  0  1  5  1]
    #      [ 1  0  1 39  0  0  0  0  0  0]
    #      [ 0  0  0  0 29  0  0  0  2  2]
    #      [ 2  0  0  6  1 32  0  0  1  0]
    #      [ 0  0  0  0  0  0 39  0  5  0]
    #      [ 0  0  0  0  0  0  0 37  1  1]
    #      [ 2  0  0  0  0  0  1  0 41  1]
    #      [ 0  0  0  0  0  0  1  0  0 30]]
    # acc: 0.905
    # prec mi: 0.905
    # prec ma: 0.9177587820233122
    # prec we: 0.9171143654568913
    # rec mi: 0.905
    # rec ma: 0.9074618933824283
    # rec we: 0.905
    # f1 mi: 0.905
    # f1 ma: 0.9082366436156739
    # f1 we: 0.9064827917258333
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[16  1  4  4  3  4  3  1  1  2]
    #      [10  6  3  0  1  5  0  7  1  8]
    #      [14  2  9  2  1  5  2  2  1  7]
    #      [10  2  4  4  2  8  1  3  3  4]
    #      [11  1  4  2  1  3  1  6  0  4]
    #      [ 9  1  2  2  2 17  1  3  1  4]
    #      [12  1  7  2  0  8  0  1  0 13]
    #      [ 8  1  4  1  0  0  0 17  2  6]
    #      [10  7  7  3  0  5  0  5  2  6]
    #      [ 8  2  3  2  1  5  0  4  1  5]]
    # acc: 0.1925
    # prec mi: 0.1925
    # prec ma: 0.17440493207996172
    # prec we: 0.17664268849471182
    # rec mi: 0.1925
    # rec ma: 0.19308559872682612
    # rec we: 0.1925
    # f1 mi: 0.19249999999999998
    # f1 ma: 0.16713768252016045
    # f1 we: 0.1681006117987566
    #
    # CGMLVQ ({'coefficients': 1000, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[37  0  1  0  0  0  1  0  0  0]
    #      [ 0 40  0  0  0  0  0  0  0  1]
    #      [ 0  0 41  0  0  0  1  0  3  0]
    #      [ 1  0  1 38  0  1  0  0  0  0]
    #      [ 0  0  0  0 30  0  0  0  2  1]
    #      [ 1  0  0  4  1 34  1  0  0  1]
    #      [ 0  1  0  0  0  0 42  0  1  0]
    #      [ 0  0  0  0  0  0  0 39  0  0]
    #      [ 0  1  0  0  0  0  0  0 43  1]
    #      [ 0  0  0  0  2  0  0  0  1 28]]
    # acc: 0.93
    # prec mi: 0.93
    # prec ma: 0.9308201991806643
    # prec we: 0.9318537513842746
    # rec mi: 0.93
    # rec ma: 0.9294209619386644
    # rec we: 0.93
    # f1 mi: 0.93
    # f1 ma: 0.9290234266535806
    # f1 we: 0.9297586394626047
    #
    #
    # Durchgang 4: Klasse 1: 154
    #              Klasse 2: 149
    #              Klasse 3: 172
    #              Klasse 4: 161
    #              Klasse 5: 155
    #              Klasse 6: 156
    #              Klasse 7: 170
    #              Klasse 8: 160
    #              Klasse 9: 164
    #              Klasse 10: 159
    #
    # kNN (k=1)
    # ---
    # cm: [[44  0  0  0  0  0  1  0  1  0]
    #      [ 0 51  0  0  0  0  0  0  0  0]
    #      [ 0  0 28  0  0  0  0  0  0  0]
    #      [ 0  0  3 35  0  0  0  1  0  0]
    #      [ 0  0  0  0 43  0  1  0  0  1]
    #      [ 0  0  0  2  0 41  0  0  1  0]
    #      [ 0  0  0  0  0  0 30  0  0  0]
    #      [ 0  0  0  0  0  0  0 38  0  2]
    #      [ 1  0  0  0  0  0  1  0 34  0]
    #      [ 0  0  0  0  1  0  0  0  2 38]]
    # acc: 0.955
    # prec mi: 0.955
    # prec ma: 0.9509238251295894
    # prec we: 0.9564871938511581
    # rec mi: 0.955
    # rec ma: 0.9562605086677198
    # rec we: 0.955
    # f1 mi: 0.955
    # f1 ma: 0.952839061421745
    # f1 we: 0.9551114187580468
    #
    # kNN (k=40)
    # ---
    # cm: [[43  0  0  0  0  0  0  0  2  1]
    #      [ 1 49  0  0  0  0  0  0  1  0]
    #      [ 0  0 26  0  0  0  0  0  2  0]
    #      [ 0  0  2 34  0  0  0  0  3  0]
    #      [ 3  0  0  0 37  0  1  0  2  2]
    #      [ 0  0  1  0  0 35  0  0  8  0]
    #      [ 0  0  0  0  0  1 28  0  1  0]
    #      [ 0  0  1  0  0  0  0 31  2  6]
    #      [ 1  0  0  2  0  0  2  0 31  0]
    #      [ 0  0  0  0  1  0  0  0  3 37]]
    # acc: 0.8775
    # prec mi: 0.8775
    # prec ma: 0.8924060873367917
    # prec we: 0.900669611847045
    # rec mi: 0.8775
    # rec ma: 0.8785493459298899
    # rec we: 0.8775
    # f1 mi: 0.8775
    # f1 ma: 0.878784389035344
    # f1 we: 0.8825515835094028
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[ 6 13  7  2  5  4  1  0  0  8]
    #      [ 5 16  9  3  1  2  0  8  0  7]
    #      [ 1  9  6  1  1  1  1  3  0  5]
    #      [ 6  4  8  7  3  4  1  1  0  5]
    #      [ 9  9  4  3  8  3  2  3  0  4]
    #      [ 4  3  6  5  6 10  6  0  0  4]
    #      [ 7  2  6  1  3  1  4  0  0  6]
    #      [ 2  8  5  1  0  1  0 11  0 12]
    #      [ 2  6  6  1  2  3  1  7  1  7]
    #      [ 2  7 11  3  4  3  0  3  0  8]]
    # acc: 0.1925
    # prec mi: 0.1925
    # prec ma: 0.29233423167246697
    # prec we: 0.287007098293863
    # rec mi: 0.1925
    # rec ma: 0.18742167339587962
    # rec we: 0.1925
    # f1 mi: 0.19249999999999998
    # f1 ma: 0.18557141373428077
    # f1 we: 0.1908472573636526
    #
    # CGMLVQ ({'coefficients': 1000, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[43  0  0  0  0  0  1  0  1  1]
    #      [ 0 49  0  0  0  0  0  0  2  0]
    #      [ 0  0 27  0  0  0  0  1  0  0]
    #      [ 0  0  2 35  0  0  0  0  2  0]
    #      [ 3  0  1  0 36  0  4  0  0  1]
    #      [ 0  0  1  2  0 40  0  0  1  0]
    #      [ 0  0  0  0  1  2 27  0  0  0]
    #      [ 0  0  0  0  0  0  0 37  0  3]
    #      [ 2  0  0  0  0  0  3  0 31  0]
    #      [ 0  0  0  0  1  0  0  0  2 38]]
    # acc: 0.9075
    # prec mi: 0.9075
    # prec ma: 0.9036201901707589
    # prec we: 0.912404577925946
    # rec mi: 0.9075
    # rec ma: 0.9079319822637458
    # rec we: 0.9075
    # f1 mi: 0.9075
    # f1 ma: 0.9039818674461717
    # f1 we: 0.908266719405278
    #
    #
    # Durchgang 5: Klasse 1: 164
    #              Klasse 2: 170
    #              Klasse 3: 156
    #              Klasse 4: 158
    #              Klasse 5: 163
    #              Klasse 6: 164
    #              Klasse 7: 153
    #              Klasse 8: 157
    #              Klasse 9: 162
    #              Klasse 10: 153
    #
    # kNN (k=1)
    # ---
    # cm: [[34  0  0  0  1  0  0  0  1  0]
    #      [ 0 30  0  0  0  0  0  0  0  0]
    #      [ 0  0 43  0  0  1  0  0  0  0]
    #      [ 0  0  1 40  0  0  0  0  1  0]
    #      [ 0  0  0  0 34  0  0  0  1  2]
    #      [ 0  0  0  3  0 31  0  0  2  0]
    #      [ 0  0  0  1  0  0 45  0  1  0]
    #      [ 0  0  0  0  1  0  0 40  1  1]
    #      [ 0  0  0  1  0  0  0  0 37  0]
    #      [ 0  0  0  0  0  0  0  0  1 46]]
    # acc: 0.95
    # prec mi: 0.95
    # prec ma: 0.9540353793032365
    # prec we: 0.9537991780045352
    # rec mi: 0.95
    # rec ma: 0.9494215135559962
    # rec we: 0.95
    # f1 mi: 0.9500000000000001
    # f1 ma: 0.9503528973429317
    # f1 we: 0.9505665745101859
    #
    # kNN (k=40)
    # ---
    # cm: [[35  0  0  0  0  0  0  0  1  0]
    #      [ 0 29  0  0  0  0  0  0  1  0]
    #      [ 0  0 42  1  0  0  1  0  0  0]
    #      [ 0  0  0 39  0  0  0  0  3  0]
    #      [ 0  0  0  0 32  0  0  0  2  3]
    #      [ 0  0  0  2  0 32  0  0  2  0]
    #      [ 1  0  0  0  1  0 39  0  6  0]
    #      [ 0  0  0  0  1  0  0 36  2  4]
    #      [ 2  0  0  0  0  0  0  0 36  0]
    #      [ 0  0  0  0  1  0  0  0  3 43]]
    # acc: 0.9075
    # prec mi: 0.9075
    # prec ma: 0.9241766917293232
    # prec we: 0.9241500939849625
    # rec mi: 0.9075
    # rec ma: 0.9105018100201567
    # rec we: 0.9075
    # f1 mi: 0.9075
    # f1 ma: 0.9124877281652701
    # f1 we: 0.9109946400440146
    #
    # CGMLVQ ({'coefficients': 0, 'totalsteps': 50, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[ 3  4 12  1  7  4  0  3  1  1]
    #      [ 4  5  5  0  3  5  0  7  1  0]
    #      [ 6  4 17  0  4  4  2  5  0  2]
    #      [ 2  7  7  2 11 11  0  0  1  1]
    #      [ 3  8  3  0  9 10  2  1  0  1]
    #      [ 4  5  6  3  4 12  2  0  0  0]
    #      [ 5 10  6  1 10  8  2  3  2  0]
    #      [ 0  8 10  0  8  1  1 11  2  2]
    #      [ 2  7  9  0  7  5  1  3  3  1]
    #      [ 2  8  8  1  5 10  1  6  4  2]]
    # acc: 0.165
    # prec mi: 0.165
    # prec ma: 0.18092877371746166
    # prec we: 0.18638412661178771
    # rec mi: 0.165
    # rec ma: 0.16804269654474085
    # rec we: 0.165
    # f1 mi: 0.165
    # f1 ma: 0.1462097360823217
    # f1 we: 0.14610758592500642
    #
    # CGMLVQ ({'coefficients': 1000, 'totalsteps': 200, 'doztr': True, 'mode': 1, 'mu': 0, 'rndinit': False})
    # ------
    # cm: [[36  0  0  0  0  0  0  0  0  0]
    #      [ 0 30  0  0  0  0  0  0  0  0]
    #      [ 0  0 40  2  0  0  1  1  0  0]
    #      [ 1  0  0 38  0  0  0  0  3  0]
    #      [ 0  0  0  0 35  0  1  0  0  1]
    #      [ 0  0  0  2  0 33  1  0  0  0]
    #      [ 1  0  0  0  1  1 44  0  0  0]
    #      [ 0  0  0  0  1  0  0 39  1  2]
    #      [ 1  0  1  3  0  1  0  0 31  1]
    #      [ 0  0  0  0  0  0  0  0  1 46]]
    # acc: 0.93
    # prec mi: 0.93
    # prec ma: 0.9324215536299085
    # prec we: 0.93113586132702
    # rec mi: 0.93
    # rec ma: 0.931412526135696
    # rec we: 0.93
    # f1 mi: 0.93
    # f1 ma: 0.9312483794709514
    # f1 we: 0.9298775114759351